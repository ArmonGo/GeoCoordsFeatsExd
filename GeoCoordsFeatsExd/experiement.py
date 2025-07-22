
from model import * 
from sklearn.metrics import root_mean_squared_error
from utility import GridSearcher
from dataloader import * 
import time 
from itertools import product
import torch
import os 

class Experiment:
    def __init__(self, result_save_p, ml_params = None, transformer_params = None):
        self.result_save_p = result_save_p
        self.ml_params = ml_params
        self.transformer_params = transformer_params
        # result log
        self.datalist = ['anemones', 'bronzefilter', 'longleaf', 'spruces', 'waka']
        self.p = 'YOUR OWN DATA PATH FOR COORDINATES ONLY'
        self.xy_data_paths = [self.p + i + '.csv' for i in self.datalist]
        self.dataload_ls = [ load_singapore, load_london, load_melbourne, load_newyork, load_paris, load_beijing, load_perth, load_seattle, load_dubai, load_yield]
        self.data_n = [  'singapore', 'london', 'melbourne', 'newyork', 'paris', 'beijing', 'perth', 'seattle', 'dubai', 'yield']
       
        self.rst = {i :{} for i in self.data_n}
        self.rst.update({i + '_coords': {} for i in self.data_n})
        self.rst.update({i + '_coords': {} for i in self.datalist})

    def load_data(self, loader = None, path = None, coords_only = False):
        if loader == load_xy_only_data:
            df = loader(path, split_rate=(0.7, 0.1, 0.2), scale =True)
        else:
            df = loader(split_rate=(0.7, 0.1, 0.2), scale =True, coords_only = coords_only)
        return df 

    def  ml_run(self, df_label, df):
        score_f = root_mean_squared_error
        # prepare data 
        df_train_raw = df[df['split_type'] == 0].sample(frac=1).reset_index(drop=True)
        y_train = df_train_raw['regression_target']
        df_train_raw = df_train_raw.drop(['regression_target', 'split_type'], axis=1)

        df_val_raw = df[df['split_type'] == 1].sample(frac=1).reset_index(drop=True)
        y_val = df_val_raw['regression_target']
        df_val_raw = df_val_raw.drop(['regression_target', 'split_type'], axis=1)

        df_test_raw = df[df['split_type'] == 2]
        y_test = df_test_raw['regression_target']
        df_test_raw = df_test_raw.drop(['regression_target', 'split_type'], axis=1)
        train_size = len(df_train_raw)
        data_size = len(df)
        for k, params in self.ml_params.items():
            print(k)
            regressor = params['regressor']
            searching_param = params['searching_param']
            search_num = 1
            # dataset 
            if k in ['GWR', 'Kriging', 'Kriging_LGBMRegressor']:
                df_train = df_train_raw.drop(['x', 'y'], axis=1)
                df_val = df_val_raw.drop(['x', 'y'], axis=1)
                df_test = df_test_raw.drop(['x', 'y'], axis=1)
            else:
                df_train = df_train_raw.drop(['lat', 'lon'], axis=1)
                df_val = df_val_raw.drop(['lat', 'lon'], axis=1)
                df_test = df_test_raw.drop(['lat', 'lon'], axis=1)

            s_t = time.time()
            try:
                if searching_param is not None:
                    searcher = GridSearcher(searching_param, score_f)
                    if k not in ['LightGBM', 'XGBoost', 'CatBoost']:
                        best_score, best_param, best_model, search_num = searcher.search(regressor, df_train, df_val, y_train, y_val)
                    else:
                        best_score, best_param, best_model, search_num = searcher.search_tree_ensemble(regressor, df_train, df_val, y_train, y_val, choice = k)
                else:
                    regressor.fit(df_train, y_train)
                    best_model = regressor
                    best_param = None 
                e_t = time.time()
                pred = best_model.predict(df_test)
                rst = score_f(y_test, pred)
                self.rst[df_label][k] = (pred, y_test, rst, best_param, e_t - s_t, search_num, train_size, data_size)
                print(k, rst)
            except Exception as e:
                print(f"An error occurred: {e}")
                self.rst[df_label][k] = e # log error message 
                continue
        return 'ml done!'
    
    def run_transformer(self, df_label, df):
        score_f = root_mean_squared_error
        # prepare data 
        df_train_raw = df[df['split_type'] == 0].sample(frac=1).reset_index(drop=True)
        y_train = df_train_raw['regression_target']
        df_train = df_train_raw.drop(['split_type', 'lat', 'lon'], axis=1)

        df_val_raw = df[df['split_type'] == 1].sample(frac=1).reset_index(drop=True)
        y_val = df_val_raw['regression_target']
        df_val = df_val_raw.drop(['split_type', 'lat', 'lon'], axis=1)

        df_test_raw = df[df['split_type'] == 2]
        y_test = df_test_raw['regression_target']
        df_test = df_test_raw.drop(['split_type', 'lat', 'lon', 'regression_target'], axis=1)
        train_size = len(df_train_raw)
        data_size = len(df)
        
        for k, params in self.transformer_params.items():
            print(k)
            model_config = params['model_config']
            search_space = params['search_space']
            search_num = 1
            # dataset 

            s_t = time.time()
            if search_space is not None:
                search_num  = len(list(product(*search_space.values())))
                try:
                    searcher = Transformers(df_train, df_val, search_space, model_config)
                    searcher.tune()
                    best_model = copy.deepcopy(searcher.model)
                    e_t = time.time()
                    pred = best_model.predict(df_test)
                    rst = score_f(y_test, pred)
                    self.rst[df_label][k] = (pred, y_test, rst, searcher.best_param, e_t - s_t, search_num, train_size, data_size)
                    print(k, rst)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    self.rst[df_label][k] = e # log error message 
                    continue
        return 'transformer done!'
    
    def rerun_specific(self, data_name, coords_only = False):
        print(f'experiment for data {data_name} begins!')
        ix = self.data_n.index(data_name)
        loader = self.dataload_ls[ix]
        df = self.load_data(loader, coords_only = coords_only)
        if coords_only:
            self.ml_run(self.data_n[ix] + '_coords', df)
            self.run_transformer(self.data_n[ix] + '_coords', df)
        else:
            self.ml_run(self.data_n[ix], df)
            self.run_transformer(self.data_n[ix], df)
        return 'done'


    def main(self, resume = None, coords_only = False, resave = True):
        print('experiment begins!')
        if resume is not None:
            self.rerun_specific(data_name=resume, coords_only = coords_only)
        else:
            if coords_only is False:
                # run estate 
                for ix in range(len(self.data_n)):
                    print(self.data_n[ix], ' begins!')
                    loader = self.dataload_ls[ix]
                    df = self.load_data(loader, coords_only = False)
                    self.run_transformer(self.data_n[ix], df)
                    self.ml_run(self.data_n[ix], df)
                    
            elif coords_only is True:
                for ix in range(len(self.data_n)):
                    loader = self.dataload_ls[ix]
                    df = self.load_data(loader, coords_only = True)
                    print(self.data_n[ix] + '_coords', ' begins!')
                    self.run_transformer(self.data_n[ix]+ '_coords', df)
                    self.ml_run(self.data_n[ix] + '_coords', df)
                
                # run coords only 
                for ix in range(len(self.datalist)):
                    loader = load_xy_only_data
                    df = self.load_data(loader, path = self.xy_data_paths[ix])
                    print(self.datalist[ix] + '_coords', ' begins!')
                    self.run_transformer(self.datalist[ix] + '_coords', df)
                    self.ml_run(self.datalist[ix] + '_coords', df)

            elif coords_only is None: # run all 
                # run estate 
                for ix in range(len(self.data_n)):
                    print(self.data_n[ix], ' begins!')
                    loader = self.dataload_ls[ix]
                    df = self.load_data(loader, coords_only = False)
                    self.run_transformer(self.data_n[ix], df)
                    self.ml_run(self.data_n[ix], df)

                for ix in range(len(self.data_n)):
                    loader = self.dataload_ls[ix]
                    df = self.load_data(loader, coords_only = True)
                    print(self.data_n[ix] + '_coords', ' begins!')
                    self.run_transformer(self.data_n[ix]+ '_coords', df)
                    self.ml_run(self.data_n[ix] + '_coords', df)
                
                # run coords only 
                for ix in range(len(self.datalist)):
                    loader = load_xy_only_data
                    df = self.load_data(loader, path = self.xy_data_paths[ix])
                    print(self.datalist[ix] + '_coords', ' begins!')
                    self.run_transformer(self.datalist[ix] + '_coords', df)
                    self.ml_run(self.datalist[ix] + '_coords', df)
        
        if resave:
            file_path = self.result_save_p + 'model_performance_final.pt'
            current_running_key = [k for k in self.rst.keys() if len(list(self.rst[k].keys())) > 0 ]
            if os.path.isfile(file_path):
                before_rst = torch.load(file_path, weights_only=False)
                before_running_key = [k for k in before_rst.keys() if len(list(before_rst[k].keys())) > 0 ]
                update_keys = [x for x in before_running_key if x not in current_running_key]
                for k in update_keys:
                    self.rst[k] = before_rst[k]
        torch.save(self.rst, file_path)
        
