# data reader
import pandas as pd
import numpy as np
import copy 
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from category_encoders.cat_boost import CatBoostEncoder
import kagglehub
from shapely import Point
import osmnx as ox 
from utility import get_local_crs,find_centroid_and_radius
import geopandas
#============================================================================================
# dataload and scaling functions
#============================================================================================
eps = 1e-5

def load_data_path(path, target_tb, format = 'csv'):
    path = kagglehub.dataset_download(path)
    if format =='csv':
        df = pd.read_csv(path + '/' +target_tb,encoding_errors='ignore')
    else:
        arrays = dict(np.load(path + '/' +target_tb))
        data = {k: [s.decode("utf-8") for s in v.tobytes().split(b"\x00")] if v.dtype == np.uint8 else v for k, v in arrays.items()}
        df = pd.DataFrame.from_dict(data)
    return df 


def scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type'):
    cols = df.columns
    if scaler is not None:
        s = scaler(feature_range = (0 + eps, 1 + eps ))
    else: # default
        s = MinMaxScaler()
    if skip_coords:
        cols = [col for col in df.columns if col not in ['lon', 'lat', mask_col]]
    if mask_col is not None:
        X = df[df[mask_col] == 0][cols]
    else:
        X = df[cols]
    s.fit(X)
    df.loc[:, cols] = s.transform(df[cols])
    return df 
        
    
def train_val_test_split(split_rate, length, shuffle = False, return_type = 'feats'):
    tr_r, val_r, te_r = split_rate
    assert tr_r + val_r + te_r == 1
    if shuffle:
        indices = np.random.permutation(length)
    else:
        indices = np.arange(length)
    ix_ls = [indices[:int(tr_r*length)], indices[int(tr_r*length):int((val_r + tr_r)*length)], indices[int((val_r + tr_r)*length):]]
    if return_type == 'index':
        mask_ls = []
        for i in range(3):
            mask = np.zeros(length, dtype=bool)
            mask[ix_ls[i]] = True
            mask_ls.append(mask)
        return mask_ls
    elif return_type == 'feats':
        split_type =  np.zeros(length, dtype=int)
        for i in range(3):
            split_type[ix_ls[i]] = i # 0-train, 1-val, 2-test
        return split_type
    
def convert_coords(df):
    df['geometry'] = list(zip(df.lon, df.lat)) # xy for later projection
    df['geometry'] = df['geometry'].apply(lambda x : Point(x))
    coords = list(zip(df.lat, df.lon))
    centroid, radius  = find_centroid_and_radius(coords)
    print('centroid', centroid)
    print('radius', radius)
    to_csr = get_local_crs(centroid[0], centroid[1], radius)  
    df['geometry']  = list(map(lambda x : ox.projection.project_geometry(x, to_crs=to_csr)[0], df['geometry'] )) 
    df['y'], df['x'] = np.array(list(map(lambda x : x.y, df.geometry))), \
                       np.array(list(map(lambda x : x.x, df.geometry)))
    df['lat'], df['lon'] = np.array(list(map(lambda x : x.y, df.geometry))), \
                       np.array(list(map(lambda x : x.x, df.geometry)))
    df = df.drop(['geometry'], axis=1)
    return df 
    


#============================================================================================
# estate data 
#============================================================================================

def load_beijing( split_rate=None, scale =True, coords_only = False):
    df_raw = pd.read_csv('DATA_PATH', encoding_errors='ignore') # use chinese inside but not utf8
    df = copy.deepcopy(df_raw[[ 'square', 'livingRoom', 'drawingRoom', 'kitchen',
    'bathRoom', 'floor', 'buildingType', 'constructionTime',
    'renovationCondition', 'buildingStructure', 'elevator',
    'fiveYearsProperty']])
    df["regression_target"] = df_raw["price"] 
    df["lon"] = df_raw["Lng"] 
    df["lat"] = df_raw['Lat'] 
    df["floor"] = df.floor.str.extract('(\d+)')
    df["floor"]  = pd.to_numeric(df["floor"], errors='coerce')
    df["tradeTime"] = pd.to_numeric(df_raw["tradeTime"].str.replace('-',''), errors='coerce')
    df["constructionTime"] = pd.to_numeric(df_raw["constructionTime"].str.replace('-',''), errors='coerce')
    
    for i in df.columns:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['tradeTime']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df


def load_seattle(split_rate=None, scale = True, coords_only = False):
    df_raw = load_data_path("harlfoxem/housesalesprediction", 'kc_house_data.csv')
    df = copy.deepcopy(df_raw[['bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
    ]])
    df["regression_target"] = df_raw["price"]/df_raw["sqft_living"]
    df["lon"] = df_raw["long"] 
    df["lat"] = df_raw['lat'] 
    df["date"] = pd.to_numeric(df_raw["date"].str[:8], errors='coerce')
    for i in df.columns:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['date']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df

def load_perth( split_rate=None, scale = True, coords_only =False):
    df_raw = load_data_path("syuzai/perth-house-prices", 'all_perth_310121.csv')
    df = copy.deepcopy(df_raw[['BEDROOMS', 'BATHROOMS', 'GARAGE',
    'LAND_AREA', 'FLOOR_AREA', 'BUILD_YEAR']])

    df["regression_target"] = df_raw["PRICE"]/df_raw["FLOOR_AREA"]
    df["lon"] = df_raw["LONGITUDE"] 
    df["lat"] = df_raw['LATITUDE'] 
    df["date"] = list(map(lambda x: x[1]+x[0], df_raw['DATE_SOLD'].str.replace('\r', '').str.split('-')))
    df["date"] = pd.to_numeric(df["date"].str[:8], errors='coerce')
    df.loc[np.isnan(df["GARAGE"]), "GARAGE"] =0
    for i in df.columns:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['date']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df


def load_london( split_rate=None, scale =True, coords_only = False):
    df_raw = load_data_path("jakewright/house-price-data", 'kaggle_london_house_price_data.csv')
    df = copy.deepcopy(df_raw[['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
    'tenure', 'propertyType', 'currentEnergyRating']])
    category_feats = ['tenure', 'propertyType']
    df["regression_target"] = df_raw["history_price"]/ df_raw["floorAreaSqM"]
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude'] 
    d = {'A' : 7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1, np.nan:0}
    df['currentEnergyRating'] = df['currentEnergyRating'].map(d)
    df["history_date"] = pd.to_numeric(df_raw["history_date"].str.replace('-',''), errors='coerce')
    df = df[df['history_date'] >= 20230101] # not too old data 
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['history_date']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'regression_target'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df
    
def load_singapore( split_rate=None, scale =True, coords_only = False):
    df_raw = pd.read_csv('DATA_PATH',encoding_errors='ignore')
    df = copy.deepcopy(df_raw[[ 'floor_area_sqft', 'flat_model']])
    category_feats = ['flat_model']
    df["regression_target"] = df_raw['price_per_sqft']
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude']
    df["date"] = pd.to_numeric(df_raw["month"].str.replace('-',''), errors='coerce')
    df = df[df['date'] >= 20230101] # too old data 
    df['remaining_lease'] = (df_raw['remaining_lease_years'].astype(int) * 12 +
                            df_raw['remaining_lease_months'].astype(int))
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['date']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'regression_target'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df


def load_newyork( split_rate=None, scale =True, coords_only = False):
    df_raw = load_data_path("nelgiriyewithana/new-york-housing-market", 'NY-House-Dataset.csv')
    df = copy.deepcopy(df_raw[['BEDS', 'BATH', 'PROPERTYSQFT', 'TYPE']])
    category_feats = ['TYPE']
    df["regression_target"] = df_raw['PRICE']/df_raw['PROPERTYSQFT']
    df["lon"] = df_raw["LONGITUDE"] 
    df["lat"] = df_raw['LATITUDE']
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sample(frac=1).reset_index(drop=True) # random to shuffle the table because no temporal info
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'regression_target'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df
    
def load_paris( split_rate=None, scale =True, coords_only = False):
    df_raw =  load_data_path("benoitfavier/immobilier-france", 'transactions.npz', format='npz')
    df_raw = df_raw[df_raw['ville'].str.startswith("PARIS ")]
    df_raw['date_transaction'] = df_raw['date_transaction'].dt.strftime('%Y-%m-%d')
    df_raw = df_raw[df_raw['date_transaction'] >= '2023-01-01']
    
    df = copy.deepcopy(df_raw[['date_transaction', 'type_batiment','n_pieces',
       'surface_habitable']])
    category_feats = ['type_batiment']
    df["regression_target"] = df_raw['prix']/df_raw['surface_habitable']
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude']
    df["date_transaction"] = pd.to_numeric(df_raw["date_transaction"].str.replace('-',''), errors='coerce')
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['date_transaction']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'regression_target'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df
    
def load_melbourne( split_rate=None, scale =True, coords_only = False):
    df_raw =  load_data_path("dansbecker/melbourne-housing-snapshot", 'melb_data.csv')
    df_raw = df_raw[df_raw['Landsize'] > 0]
    df = copy.deepcopy(df_raw[['Rooms', 'Type',  'Method',
       'Date', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt']])
    category_feats = ['Type','Method']

    df["regression_target"] = df_raw['Price']/df_raw['Landsize']
    df["lon"] = df_raw["Longtitude"] 
    df["lat"] = df_raw['Lattitude']
    df["Date"] = pd.to_numeric(pd.to_datetime(df_raw['Date'], dayfirst = True).dt.strftime('%Y-%m-%d').str.replace('-',''), errors='coerce')
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['Date']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'regression_target'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df
   
def load_dubai( split_rate=None, scale =True, coords_only = False):
    df_raw =  load_data_path("azharsaleem/real-estate-goldmine-dubai-uae-rental-market", 'dubai_properties.csv')
    df_raw = df_raw[df_raw['Posted_date'] > '2023-01-01']
    df = copy.deepcopy(df_raw[[ 'Beds', 'Baths', 'Type', 'Area_in_sqft', 'Furnishing', 'Age_of_listing_in_days']])
    df['Furnishing']= df['Furnishing'].map({'Unfurnished': 0 , 'Furnished': 1})
    category_feats = ['Type']

    df["regression_target"] = df_raw['Rent_per_sqft']
    df["lon"] = df_raw["Longitude"] 
    df["lat"] = df_raw['Latitude']
    df["Posted_date"] = pd.to_numeric(df_raw['Posted_date'].str.replace('-',''), errors='coerce')
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['Posted_date']) # temporal split 
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'regression_target'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df
    

#============================================================================================
# x, y only dataset
#============================================================================================

# data used: x, y and marks (regression)
# datalist = ['anemones', 'bronzefilter', 'longleaf', 'spruces', 'waka']
def load_xy_only_data(path, split_rate=None, scale =True, coords_only = False):
    df = pd.read_csv(path)
    df = df.rename(columns = {'marks': 'regression_target'})
    df['lat'] = df['y'].copy()
    df['lon'] = df['x'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat', 'x', 'y'] + [col for col in df.columns if col not in ['lat', 'lon', 'x', 'y']]]
    df = df.astype({"lon": np.float64, "lat": np.float64})
    return df
    


#============================================================================================
# others : crime, yield, earning
#============================================================================================


def load_yield(split_rate=None, scale =True, coords_only = False):
    df = pd.read_csv('./Dataset/extra_feats/rosas2001.csv')
    df = df.rename(columns = {'YIELD': 'regression_target'})
    df["lon"] = df["LONGITUDE"] 
    df["lat"] = df['LATITUDE']
    df = df.drop(columns=['ID', 'LONGITUDE', 'LATITUDE'])
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sample(frac=1).reset_index(drop=True) # random to shuffle the table because no temporal info
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df
    


def load_earning(split_rate=None, scale =True, coords_only = False):
    df = geopandas.read_file('./Dataset/extra_feats/NYC Area2010_2data.dbf')
    df = df.rename(columns = {'C000_13': 'regression_target'})
    df = df[['UR10', 'AWATER10', 'INTPTLAT10', 'INTPTLON10',
       'Shape_area', 'Shape_len', 'regression_target']]
    d = {'U' : 0, 'R': 1}
    df['UR10'] = df['UR10'].map(d)
    df["lon"] = df["INTPTLON10"] 
    df["lat"] = df['INTPTLAT10']
    df.lon = df.lon.astype(float)
    df.lat = df.lat.astype(float)
    df = df.drop(columns=['INTPTLON10', 'INTPTLAT10'])
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sample(frac=1).reset_index(drop=True) # random to shuffle the table because no temporal info
    df = convert_coords(df)
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    # df = df.astype({col: np.float32 for col in df.select_dtypes(include=['float64']).columns})
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'regression_target', 'split_type']]
    else:
        return df
    