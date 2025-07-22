
from model import * 
from dataloader import * 
import torch

class FitBest:
    def __init__(self, result_save_p, loader = None):
        self.result_save_p = result_save_p
        self.loader = loader  
        self.load_data()

    def load_data(self):
        df = self.loader(split_rate=(0.7, 0.1, 0.2), scale =True, coords_only = False)
        # prepare machining learning data 
        df_train_raw_ml = df[df['split_type'] == 0].sample(frac=1).reset_index(drop=True)
        y_train_ml = df_train_raw_ml['regression_target']
        df_train_raw_ml = df_train_raw_ml.drop(['regression_target', 'split_type'], axis=1)

        df_test_raw_ml = df[df['split_type'] == 2]
        y_test_ml = df_test_raw_ml['regression_target']
        df_test_raw_ml = df_test_raw_ml.drop(['regression_target', 'split_type'], axis=1)
        self.ml_data = (df_train_raw_ml, y_train_ml, df_test_raw_ml, y_test_ml)

        # prepare transformer data 
        df_train_raw_t = df[df['split_type'] == 0].sample(frac=1).reset_index(drop=True)
        df_train_t = df_train_raw_t.drop(['split_type', 'lat', 'lon'], axis=1)

        df_val_raw_t = df[df['split_type'] == 1].sample(frac=1).reset_index(drop=True)
        df_val_t = df_val_raw_t.drop(['split_type', 'lat', 'lon'], axis=1)

        df_test_raw_t = df[df['split_type'] == 2]
        y_test_t = df_test_raw_t['regression_target']
        df_test_t = df_test_raw_t.drop(['split_type', 'lat', 'lon', 'regression_target'], axis=1)
        self.transform_data = (df_train_t, df_val_t, df_test_t, y_test_t)
        return self.ml_data, self.transform_data
    
    def ml_refit(self, m_name, m_inst, df_label):
        (X_train, y_train, X_test, y_test) = copy.deepcopy(self.ml_data)
        if m_name in ['GWR', 'Kriging', 'Kriging_LGBMRegressor']:
            X_train = X_train.drop(['x', 'y'], axis=1)
            X_test = X_test.drop(['x', 'y'], axis=1)
        else:
            X_train = X_train.drop(['lat', 'lon'], axis=1)
            X_test = X_test.drop(['lat', 'lon'], axis=1)
        best_params = torch.load(self.result_save_p  + 'model_performance_final.pt', weights_only=False)[df_label][m_name][3]
        if best_params is not None:
            m_inst = m_inst(**best_params)
        m_inst.fit(X_train, y_train)
        return m_inst, X_train, X_test, y_test
    
    def transformer_refit(self, m_name, m_config, df_label):
        (X_train, X_val, X_test, y_test) = copy.deepcopy(self.transform_data)
        best_params = torch.load(self.result_save_p  + 'model_performance_final.pt', weights_only=False)[df_label][m_name][3]
        
        for n, v in best_params.items():
            if 'model_config__' in n:
                m_config.__setattr__(n.split('model_config__')[1], v)
        print(m_config)
        m_inst = Transformers(X_train, X_val, None, m_config)
        m_inst.fit()
        return m_inst, X_train, X_test, y_test
