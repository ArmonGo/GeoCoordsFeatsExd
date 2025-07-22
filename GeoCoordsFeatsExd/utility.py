
import numpy as np 
import osmnx as ox 
import pyproj
from sklearn.model_selection import ParameterGrid
import lightgbm as lgb 
import xgboost as xgb 
import math 
import copy 
import matplotlib.pyplot as plt
import shap
import pandas as pd 
import torch 
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib as mpl
import os 
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

plt.style.use('seaborn-v0_8-talk') # Solarize_Light2

def get_local_crs(lat, lon, radius):  
    trans = ox.utils_geo.bbox_from_point((lat, lon), dist = radius, project_utm = True, return_crs = True)
    to_csr = pyproj.CRS( trans[-1])
    return to_csr


# Function to calculate the distance between two geographic coordinates using the Haversine formula
def haversine(coord1, coord2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Function to find the centroid and radius
def find_centroid_and_radius(coords):
    # Calculate the centroid (average latitude and longitude)
    avg_lat = sum(coord[0] for coord in coords) / len(coords)
    avg_lon = sum(coord[1] for coord in coords) / len(coords)
    centroid = (avg_lat, avg_lon)
    # Calculate the radius (maximum distance from centroid to any point)
    radius = max(haversine(centroid, coord) for coord in coords) * 1000
    return centroid, np.ceil(radius) + 5000 # add buffer 


class GridSearcher:
    def __init__(self, grid, score_f):
        self.param_grid =  ParameterGrid(grid)
        self.score_f = score_f
        self.best_score = np.inf
        self.best_param = None
        self.best_model = None 

    def search(self, rg, df_train, df_val, y_train, y_val):
        for param in self.param_grid:
            input_p = copy.deepcopy(param)
            regressor = rg() # instantiate the regressor 
            regressor.set_params(**input_p)
            regressor.fit(df_train, y_train)
            # count score
            pred = regressor.predict(df_val)
            s = self.score_f(y_val, pred)
            if s < self.best_score:
                self.best_score = s
                self.best_param = param
                self.best_model = copy.deepcopy(regressor)
        return self.best_score, self.best_param, self.best_model, len(list(self.param_grid))
    
    def search_tree_ensemble(self, rg, df_train, df_val, y_train, y_val, choice):
        for param in self.param_grid:
            regressor = rg() # instantiate the regressor 
            regressor.set_params(**param)
            if choice == 'LightGBM':
                callbacks = lgb.early_stopping(stopping_rounds = 100, first_metric_only = True, verbose = False, min_delta=0.0)
                regressor.fit(df_train, y_train, callbacks = [callbacks],  # use the best iteration to predict
                                eval_metric ='rmse', eval_set =[(df_val, y_val)])
            elif choice == 'XGBoost':
                callbacks = xgb.callback.EarlyStopping(rounds=100, metric_name='rmse', data_name='validation_0', save_best=True)  
                regressor.set_params(**{'callbacks' : [callbacks]})
                regressor.fit(df_train, y_train,  # return the best model
                             eval_set =[(df_val, y_val)],
                             verbose=False)
            elif choice == 'CatBoost':
                regressor.set_params(**param)
                regressor.fit(df_train, y_train,  # return the best model
                             eval_set =[(df_val, y_val)],
                             use_best_model=True,
                            early_stopping_rounds=100,
                            verbose_eval = False)
            # count score
            pred = regressor.predict(df_val)
            s = self.score_f(y_val, pred)
            if s < self.best_score:
                self.best_score = s
                self.best_param = param
                self.best_model = copy.deepcopy(regressor)
        return self.best_score, self.best_param, self.best_model, len(list(self.param_grid))

def load_rst(path  ='./Results/' + 'model_performance_final.pt'):
    rst = torch.load(path, weights_only=False)
    df = {
    'data_name' : [],
    'model' : [], 
    'rmse' : [], 
    'time' : [], 
    'nr.hyperparameters' : [],
    'nr.train' : [],
    'nr.df' : []
      }
    for k, v in rst.items():
        if len(v) != 0 :
            for m, p in v.items():
                try:
                    (pred, y_test, rmse, best_param, t, search_num, train_size, data_size) = p
                    df['data_name'].append(k)
                    df['model'].append(m)
                    df['rmse'].append(rmse)
                    df['time'].append(t)
                    df['nr.hyperparameters'].append(search_num)
                    df['nr.train'].append(train_size)
                    df['nr.df'].append(data_size)
                except:
                    print(k, m, p) 
                    df['data_name'].append(k)
                    df['model'].append(m)
                    df['rmse'].append(np.nan)
                    df['time'].append(np.nan)
                    df['nr.hyperparameters'].append(np.nan)
                    df['nr.train'].append(np.nan)
                    df['nr.df'].append(np.nan)
    rst_df = pd.DataFrame.from_records(df)
    rst_df['per_train_time'] = rst_df['time']/rst_df['nr.hyperparameters']
    rst_df = rst_df.round({'per_train_time': 3,
                       'rmse': 4})
    rst_df['model'] = rst_df['model'].map({'Lr_ridge' : "Ridge LR",
        'SVM' : "SVM",
        'GWR' : "GWR",
        'Kriging' :'Kriging',
        "Kriging_LGBMRegressor": "Kriging LGBM", 
        "Guassian":'Gaussian P', 
        'XGBoost' : "XGBoost", 
        'CatBoost' :"CatBoost",
        "TweedieRegressor": "Tweedie", 
        "RandomForest": "RMF", 
        "LightGBM": "LGBM", 
        "TabPFNRegressor": "TabPFN", 
        'FTTransformer' : 'FTTransformer', 
        'DANet': 'DANet',
        'GANDALF' : 'GANDALF', 
        'GatedAdditiveTreeEnsemble' : 'GATE',
        'Node' : 'NODE' 
        })
    
    df = pd.DataFrame(rst_df)
    df = df[~df['data_name'].str.contains('earning')]
    return df

def load_best_rmse(df, best = True):
    pivot = df.pivot(index='data_name', columns='model', values='rmse')
    if best :
        min_rmse = pivot.min(axis=1)
        # Mark the minimum RMSE value in each row
        pivot_highlighted = pivot.apply(lambda row: [
            f"*{val}*" if val == min_rmse[row.name] else val for val in row], axis=1)
        # Convert back to DataFrame for better display
        result = pd.DataFrame(pivot_highlighted.tolist(), columns=pivot.columns, index=pivot.index).reset_index()
        # Desired column order
        desired_order = ['data_name', 'Ridge LR', 'SVM', 'GWR', 'Kriging', 'Kriging LGBM',
                        'Gaussian P', 'Tweedie', 'RMF', 'LGBM', 'XGBoost', 'CatBoost', 
                        'TabPFN', 'FTTransformer', 'DANet', 'GANDALF', 'GATE', 'NODE']
        # Reorder columns based on the desired order
        rmse_table = result[[col for col in desired_order if col in result.columns] +
                                [col for col in result.columns if col not in desired_order]]
        rmse_table.columns.name = None
        result_coords = rmse_table[rmse_table['data_name'].str.contains('_coords')]
        result_all = rmse_table[~rmse_table['data_name'].str.contains('_coords')]
        return result_all, result_coords
    else:
        result = pivot[~pivot.index.str.contains('earning')]
        # Desired column order
        desired_order = ['data_name', 'Ridge LR', 'SVM', 'GWR', 'Kriging', 'Kriging LGBM',
                        'Gaussian P', 'Tweedie', 'RMF', 'LGBM', 'XGBoost', 'CatBoost', 
                        'TabPFN', 'FTTransformer', 'DANet', 'GANDALF', 'GATE', 'NODE']
        # Reorder columns based on the desired order
        rmse_table = result[[col for col in desired_order if col in result.columns] +
                                [col for col in result.columns if col not in desired_order]]
        rmse_table.columns.name = None
        result_coords = rmse_table[rmse_table.index.str.contains('_coords')]
        result_all = rmse_table[~rmse_table.index.str.contains('_coords')]
        return result_all, result_coords

def load_train_time(df):
    df_plot_coords = df[df['data_name'].str.contains('_coords')].sort_values(by=['model', 'nr.train'])
    df_plot_all = df[~df['data_name'].str.contains('_coords')].sort_values(by=['model', 'nr.train'])
    return df_plot_coords, df_plot_all



def plot_train_time(df_plot, title_prefix, save = True):
    # plt.style.available
    # Custom colormap: dark blue → light blue → orange/red
    custom_cmap = LinearSegmentedColormap.from_list(
        #"custom_shap", ["#780000", "#d84689",  "#fdf0d5", "#003049", "#669bbc"]
    'custom_shap', 
    #["#001219","#005f73","#0a9396","#94d2bd","#e9d8a6","#ee9b00","#ca6702","#bb3e03","#ae2012","#9b2226"]
    ["#f6114a","#0aa0bf","#f36e98","#78b177","#9862a2","#f05006","#25998f","#fca00c"] )
    # Data preparation

    # Line styles and number of models
    line_styles = ['-', '--', '-.', ':']
    n_models = df_plot['model'].nunique()

    # Normalize for custom cmap
    norm = Normalize(vmin=0, vmax=n_models - 1)

    # Sort models by last value
    model_last_values = {
        model: df_plot[df_plot['model'] == model]['per_train_time'].iloc[-1]
        for model in df_plot['model'].unique()
    }
    sorted_models = sorted(model_last_values, key=model_last_values.get, reverse=True)

    # Clean style, no grid
    mpl.rcParams['font.family'] = 'Times New Roman'  # Change to your preferred font
    # mpl.rcParams['font.size'] = 20
    
    # Plot
    plt.figure(figsize=(20, 15), dpi = 300)
    for i, model in enumerate(sorted_models):
        model_data = df_plot[df_plot['model'] == model]
        color = custom_cmap(norm(i))
        linestyle = line_styles[i % len(line_styles)]
        
        plt.plot(model_data['nr.train'], model_data['per_train_time'],
                label=model,
                color=color,
                linestyle=linestyle,
                marker='o',
                markersize=4)

    # Log scales
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title
    #plt.title( title_prefix + ' Datasets: Training Time vs. Train Data Size (Log-Log Scale)', fontsize=30)
    plt.xlabel('Train Data Size (log scale)', fontsize=25)
    plt.ylabel('Train Time (log scale)', fontsize=25)

    # Legend
    plt.legend(title="",
               # bbox_to_anchor=(0.5, -0.1),
                loc='upper left', fontsize=24, ncol=4)

    # Remove grid and tidy layout
    plt.grid(False)
    if save :
        plt.savefig(title_prefix + ' taining time'+ '.png')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


## =======================================================================================
## interpretation 
## =======================================================================================


def transformer_explain(transformer, df_test, n = 100, random_state = 42, mean = True):
    df_sample = df_test.sample(n=n, random_state=random_state)
    importance = transformer.model.explain(df_sample, method="GradientShap", baselines="b|1000")
    if mean:
        importance = importance.mean().to_frame().T
    return importance, df_sample


def ml_explain(model, df_train, df_test, n = 100, random_state = 42, mean = True):
    df_sample = df_test.sample(n=n, random_state=random_state).reset_index(drop =True)
    explainer = shap.Explainer(model, df_train)  # SHAP chooses the best explainer under the hood
    shap_values_raw = explainer(df_sample)
   
    shap_values = pd.DataFrame( shap_values_raw.values, columns=df_train.columns)
    if mean:
        shap_values_mean = shap_values.mean().to_frame().T
    return shap_values_mean, shap_values, shap_values_raw

def plot_shap_values(model_info_ls, file_name):
    # Set global font
    mpl.rcParams['font.family'] = 'Times New Roman'  # Change to your preferred font
    mpl.rcParams['font.size'] = 15    

    # Create custom blue-to-orange colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_shap", ["#2e4057","#bfdbf7", "#a31621"]
    )

    # Prepare SHAP plots
    os.makedirs("shap_temp", exist_ok=True)

    n_models = len(model_info_ls)
    ncols = 4
    nrows = math.ceil(n_models / ncols)

    for i, (title, shap_values, X_input) in enumerate(model_info_ls):
        plt.figure(figsize=(16, 8), dpi=300)
        shap.summary_plot(
            shap_values,
            X_input,
            plot_type="dot",
            show=False,
            max_display=10,
            cmap=custom_cmap
        )

        # Change font of all text elements in the current SHAP plot
        for ax in plt.gcf().axes:
            for label in ax.get_xticklabels():
                label.set_fontsize(15)
                label.set_fontname('Times New Roman')
                label.set_rotation(45)
                label.set_ha('right')
            for label in ax.get_yticklabels():
                label.set_fontsize(26)
                label.set_fontname('Times New Roman')
            ax.title.set_fontsize(22)
            ax.title.set_fontname('Times New Roman')
            ax.xaxis.label.set_fontsize(24)
            ax.yaxis.label.set_fontsize(24)
            ax.xaxis.label.set_fontname('Times New Roman')
            ax.yaxis.label.set_fontname('Times New Roman')
            if 'colorbar' in ax.get_label().lower():
                ax.remove()

        plt.tight_layout()
        plt.savefig(f"shap_temp/{i:02d}_{title}.png")
        plt.close()

    # Combine saved images into subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=( 6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (title, _, _) in enumerate(model_info_ls):
        img = plt.imread(f"shap_temp/{i:02d}_{title}.png")
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(title.replace("_", " "), fontsize=20)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig( file_name, bbox_inches='tight')
    plt.show()



## =======================================================================================
## Demšar Analysis 
## =======================================================================================

def demsar_analysis(rst, plot = None, title_prefix = '', font_size = 15):
    # ranking table 
    ranks = rst.rank(axis=1, method='average', ascending=True) 
    
    # Convert ranks to list of columns
    friedman_result = friedmanchisquare(*[ranks[model] for model in ranks.columns])
    print("Friedman test statistic:", friedman_result.statistic)
    print("p-value:", friedman_result.pvalue)
    nemenyi = sp.posthoc_nemenyi_friedman(ranks.values)
    nemenyi.columns = rst.columns
    nemenyi.index = rst.columns
    # Set global font
    mpl.rcParams['font.family'] = 'Times New Roman'  # Change to your preferred font
    mpl.rcParams['font.size'] = font_size
    if plot is None:
        return nemenyi
    elif plot == 'CD':    

        avg_ranks = np.mean(ranks, axis=0)
        plt.figure(figsize=(10, 5), dpi=300)
        cmap = ['1', "#81a3de",  "#e05634",  "#df745c", "#f1bcae"]
        heatmap_args = {'g': list(avg_ranks.index), 'cmap': cmap, 'linewidths': 0, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        sp.sign_plot(nemenyi, **heatmap_args)
        plt.title(title_prefix + "Critical Difference Diagram", loc='left')
        plt.savefig(title_prefix + 'CD diagram.png', bbox_inches='tight')

        
    elif plot == 'rank':
        avg_ranks = np.mean(ranks, axis=0)
        plt.figure(figsize=(10, 2), dpi=300)
        plt.title(title_prefix + 'Critical difference diagram of average score ranks')
        sp.critical_difference_diagram(
            ranks=avg_ranks,
            sig_matrix=nemenyi,
            label_fmt_left='{label} [{rank:.3f}]  ',
            label_fmt_right='  [{rank:.3f}] {label}',
            text_h_margin=0.3,
            label_props={'fontweight': 'bold'},
            crossbar_props={'color': None, 'marker': 'o'},
            marker_props={'marker': '*', 's': 150, 'color': 'y', 'edgecolor': 'k'},
            elbow_props={'color': 'gray'},
        )
        
        plt.savefig(title_prefix + 'CD rank.png',bbox_inches='tight')