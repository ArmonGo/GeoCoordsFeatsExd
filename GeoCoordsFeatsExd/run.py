from experiement import Experiment
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from model import GWR, KrigeRegressor, OrdinaryKriging
import numpy as np 
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import TweedieRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.models import (
    FTTransformerConfig,
    GANDALFConfig,
    DANetConfig,
    GatedAdditiveTreeEnsembleConfig,
    NodeConfig
)


# catboost 

result_save_p = './Results/'

ml_params = {'GWR' : {'regressor' : GWR, 
                        'searching_param' : {"constant": [True]}  
                         },
                'Kriging_LGBMRegressor' : {'regressor' : KrigeRegressor, 
                             'searching_param' :   {"outer_params" : list(ParameterGrid([{ "nlags" : range(30,150,30),  #range(20,120,20),
                                                                                            "variogram_model": [ "linear", "gaussian"] ,
                                                                                            'regression_model': [LGBMRegressor]}])),  # outer model params
                                                    'inner_params' : list(ParameterGrid([ { "reg_alpha" : np.arange(0, 1.5, 0.5),
                                                                                            "reg_lambda" : np.arange(0, 1.5, 0.5),
                                                                                            "learning_rate" : [0.1, 0.01, 0.005],
                                                                                            "verbose": [-100]
                                                                                            } ]))
                                                    } 
                       },
                "Kriging": { 'regressor' : OrdinaryKriging, 
                            "searching_param": {"nlags" : range(30,150,30),
                                                "variogram_model":[ "gaussian", "linear"] } #  "spherical", "power"
                                },
                'Lr_ridge' : {'regressor' : Ridge, 
                        'searching_param' : {"alpha": np.arange(0.1,1,0.1)}
                        },
                'RandomForest' : {'regressor' : RandomForestRegressor, 
                        'searching_param' :  {
                                              "min_samples_split" : [2,3,5],
                                              "min_samples_leaf" : [3,5,10]
                                             }
                                 },     
                'XGBoost' : {'regressor' : XGBRegressor, 
                        'searching_param' :  { "learning_rate" : [0.1, 0.01, 0.005],
                                                "reg_alpha" : np.arange(0, 1.1, 0.1),
                                                "reg_lambda" : np.arange(0, 1.1, 0.1)
                                            } 
                                 },
                'LightGBM' : {'regressor' : LGBMRegressor, 
                                'searching_param' : {
                                                    "reg_alpha" : np.arange(0, 1.1, 0.1),
                                                    "reg_lambda" : np.arange(0, 1.1, 0.1),
                                                    "learning_rate" : [0.1, 0.01, 0.005],
                                                    "verbose": [-100]
                                                    }     
                             },
                
                "SVM":  { 'regressor' : SVR, 
                            "searching_param": {"C": range(1,105,10), # change the param from 5 to 10
                               "epsilon": np.arange(0.1, 1, 0.1)}
                        },
                "Guassian":  { 'regressor' : GaussianProcessRegressor, 
                            "searching_param": {"kernel":  [ C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))],
                               #  "n_restarts_optimizer": range(5, 25, 5), 
                               "alpha" : np.arange(0.1, 1, 0.1)
                                                }
                                },         
                "TweedieRegressor":  { 'regressor' : TweedieRegressor, 
                            "searching_param":  {'power' : [0, 1,1.2, 1.5, 1.8, 2, 3],
                                                  'alpha' :  list(np.arange(0, 1.0, 0.1)) + [2, 5, 8, 10]}
                                },
                "CatBoost" :  {'regressor' : CatBoostRegressor, 
                            "searching_param":  { 'iterations': [100, 200],
                                                'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
                                                # 'depth': [2, 4, 6],
                                                'l2_leaf_reg': [0.1, 0.5, 1, 5]}},
                "TabPFNRegressor" : {'regressor': TabPFNRegressor(ignore_pretraining_limits = True),
                                       "searching_param":  None
                                      
                }
}

#### for transformers 

head_config = LinearHeadConfig(
    layers="",  # No additional layer in head, just a mapping layer to output_dim
    dropout=0.1,
    initialization="kaiming",
).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)


transformer_params = {'FTTransformer': {'search_space' :  {
                                                "model_config__num_heads": [4, 8, 16],
                                                "model_config__attn_dropout": [0.1, 0.2, 0., 0.4],
                                               
                                        }, 
                                        'model_config' : FTTransformerConfig(
                                                task="regression",
                                                learning_rate=1e-3,
                                                head="LinearHead",  # Linear Head
                                                head_config=head_config,  # Linear Head Config
                                                )
                                        }, 
                        'DANet': {'search_space' :  {
                                                "model_config__n_layers" :  [8, 20],
                                                        "model_config__k":  [3, 5, 8],
                                                        "model_config__dropout_rate": [0.1, 0.2, 0.3]
                                                }, 
                                                'model_config' : DANetConfig(
                                                        task="regression",
                                                        learning_rate=1e-3,
                                                        head="LinearHead",  # Linear Head
                                                        head_config=head_config,  # Linear Head Config
                                                        )
                                                }, 
                         'GANDALF': {'search_space' :  {
                                                "model_config__gflu_stages" :  [2, 4, 6, 8, 10],
                                                "model_config__gflu_dropout": [0, 0.1, 0.2, 0.3]
                                                }, 
                                                'model_config' :  GANDALFConfig(
                                                                task="regression",
                                                                learning_rate=1e-3,
                                                                head="LinearHead",  # Linear Head
                                                                head_config=head_config,  # Linear Head Config
                                                                )
                                                },
                        'GatedAdditiveTreeEnsemble': {'search_space' :  {
                                                "model_config__gflu_stages" :  [2, 4, 6, 8, 10],
                                                "model_config__gflu_dropout": [0, 0.1, 0.2, 0.3]
                                                }, 
                                                'model_config' : GatedAdditiveTreeEnsembleConfig(
                                                                task="regression",
                                                                learning_rate=1e-3,
                                                                head="LinearHead",  # Linear Head
                                                                head_config=head_config,  # Linear Head Config
                                                                )
                                                },
                        'Node': {'search_space' :  {
                                                "model_config__num_layers" :  [1, 2, 4],
                                                "model_config__num_trees": [8, 16, 32, 64],
                                                "model_config__depth" : [3, 4, 6],
                                                "model_config__input_dropout" : [0, 0.1, 0.2, 0.3]
                                                }, 
                                                'model_config' : NodeConfig(
                                                                task="regression",
                                                                learning_rate=1e-3,
                                                                head="LinearHead",  # Linear Head
                                                                head_config=head_config,  # Linear Head Config
                                                                )
                                                }
                                          
                      }

if __name__ == "__main__":
    exp = Experiment(result_save_p, ml_params, transformer_params)
    exp.main( resume = None, coords_only = True, resave = True)
