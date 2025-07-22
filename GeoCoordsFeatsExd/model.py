# models that includes geo autocorrelation 
from pykrige.rk import RegressionKriging, Krige
import numpy as np
from mgwr.gwr import GWR as Mod_GWR
from mgwr.sel_bw import Sel_BW
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import  check_is_fitted
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig
    )
from pytorch_tabular.models.common.heads import LinearHeadConfig
import warnings
from pytorch_tabular.tabular_model_tuner import TabularModelTuner
from pytorch_tabular import TabularModel

class KrigeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, outer_params=None, inner_params=None):
        self.outer_params = outer_params
        self.inner_params = inner_params

    def split_coords(self, X, cols = [0,1]): # default of coordinates
        coords = X.iloc[:, cols]
        X = X.iloc[:, max(cols)+1: ]
        if X.shape[1] == 0:
            X = np.ones((coords.shape[0], 1))
        return np.array(X), np.array(coords)
    
    def fit(self, X, y):
        X, coords = self.split_coords(X)
        self.outer_params['regression_model'] = self.outer_params['regression_model']().set_params(**self.inner_params)
        self.pykridge_model_ = RegressionKriging(**self.outer_params)
        self.pykridge_model_.fit(X,coords,y)
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X, coords = self.split_coords(X)
        predictions = self.pykridge_model_.predict(X, coords)
        return predictions
    
class OrdinaryKriging(BaseEstimator, RegressorMixin):
    def __init__(self, nlags=5, variogram_model='gaussian'):
        self.nlags = nlags
        self.variogram_model = variogram_model

    def split_coords(self, X, cols = [0,1]): # default of coordinates
        coords = X.iloc[:, cols]
        X = X.iloc[:, max(cols)+1: ]
        if X.shape[1] == 0:
            X = np.ones((coords.shape[0], 1))
        return X, coords
    
    def fit(self, X, y):
        X, coords = self.split_coords(X)
        self.model_ = Krige(nlags=self.nlags, variogram_model = self.variogram_model)
        self.model_.fit(np.array(coords),np.array(y))
        return self
    
    def predict(self, X):
        X, coords = self.split_coords(X)
        predictions = self.model_.predict(np.array(coords))
        return predictions



class GWR(BaseEstimator, RegressorMixin):
    def __init__(self, constant=True, kernel='gaussian', bw=None):
        self.constant = constant
        self.kernel = kernel
        self.bw = bw
        
    def split_coords(self, X, cols = [0,1]): # default of coordinates
        coords = X.iloc[:, cols]
        X = X.iloc[:, max(cols)+1: ]
        if X.shape[1] == 0:
            X = np.ones((coords.shape[0], 1))
        rng = np.random.RandomState(42)
        rand = rng.randn(X.shape[0], X.shape[1])/10000 # to prevent a singular matrix
        return np.array(X+rand), np.array(coords)
    
    def fit(self, X, y):
        X, coords = self.split_coords(X)
        y = np.array(y).reshape((-1, 1))
        if self.bw is None:
            self.bw = Sel_BW(coords, y, X).search()
        self.model_ = Mod_GWR(coords, y, X, self.bw, constant=self.constant, kernel=self.kernel)
        gwr_results = self.model_.fit()
        self.scale = gwr_results.scale
        self.residuals = gwr_results.resid_response 
        return self

    def predict(self, X):
        check_is_fitted(self)
        X, coords = self.split_coords(X)
        pred = self.model_.predict(coords, X, exog_scale=self.scale, exog_resid=self.residuals
               ).predictions
        return pred
    

class Transformers:
    def __init__(self, df_train, df_val, search_space, model_config):
        # initlize configs 
        cols = list(df_train.columns)
        cols.remove('regression_target')
        self.data_config = DataConfig(
                    target=[
                        'regression_target'
                    ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
                    continuous_cols = cols,
                    categorical_cols=[]
                    )
        # trainer config 
        self.trainer_config = TrainerConfig(
            auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
            batch_size=1024,
            max_epochs=3000,
            early_stopping="valid_loss",  # Monitor valid_loss for early stopping
            early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
            checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
            load_best=True,  # After training, load the best checkpoint
            progress_bar="none",  # Turning off Progress bar
            trainer_kwargs=dict(enable_model_summary=False),  # Turning off model summary
        )
        self.optimizer_config = OptimizerConfig()

        self.head_config = LinearHeadConfig(
            layers="", dropout=0.1, initialization="kaiming"  # No additional layer in head, just a mapping layer to output_dim
        ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)
        self.df_train = df_train
        self.df_val = df_val

        self.search_space = search_space
        self.model_config = model_config
        self.model = None
        self.best_param = None

    def tune(self):
        tuner = TabularModelTuner(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = tuner.tune(
                train=self.df_train,
                validation=self.df_val,
                search_space=self.search_space,
                strategy="grid_search",
                # cv=5, # Uncomment this to do a 5 fold cross validation
                metric="mean_squared_error",
                mode="min",
                progress_bar=True,
                verbose=False # Make True if you want to log metrics and params each iteration
            )
        self.model = result.best_model
        self.best_param = result.best_params
    
    def fit(self):
        self.model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
            verbose=False
)
        self.model.fit(train=self.df_train, validation=self.df_val)
    
    def predict(self, df_test):
        y_pred = self.model.predict(df_test)
        return y_pred