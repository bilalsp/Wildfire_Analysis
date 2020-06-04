import os
import sys
import json
import time
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')

DATA_PATH = "data/"

class EnvironmentalFeaturesGenerator:
    """FeaturesGenerator to generate environmental features.
    
    It generates the features like Air_Temperature('air'), Geopotential_Height('hgt'), 
    Relative_Humidity('rhum'), East_Wind/Zonal_Wind('uwnd'), North_Wind/Meridional_Wind('vwnd'),  
    Tropopause_Pressure('trpp'), Tropopause_Temperature('trpt'), Surface_Potential_Temperature('srfpt') 
    at different pressure level using NCEP data. 
    
    NCEP data link:
    https://psl.noaa.gov/data/gridded/data.ncep.html
    https://www.ncei.noaa.gov/erddap/convert/oceanicAtmosphericVariableNames.htmlTable
    
    """
    
    def __init__(self, ncep_data, ncep_sfc_data):
        self.ncep_data = ncep_data
        self.levels = [100, 150, 200, 500, 1000]
        self.ncep_data_vars = set(ncep_data[list(ncep_data)[0]]) - set(['head'])
        self.ncep_sfc_data = ncep_sfc_data
        self.ncep_sfc_data_vars = set(ncep_sfc_data[list(ncep_sfc_data)[0]]) - set(['head'])
        
    def _extract_features(self, row):
        """extract the environmental features for particular wildfire incident"""
        ncep_data = self.ncep_data
        ncep_sfc_data = self.ncep_sfc_data
        date = row['date']
        features = dict(row)
        #reduce the dimensions of ncep_data(xarray dataset) by fixing coordinates(lon,lat)
        #and then convert it to dataframe
        ncep_data = ncep_data[date.year] \
                            .sel(lon=row['longitude'], lat=row['latitude'], method='nearest') \
                            .to_dask_dataframe() \
                            .compute() \
                            .set_index(['level','time'])
        #reduce the dimensions of ncep_sfc_data(xarray dataset) by fixing coordinates(lon,lat)
        #and then convert it to dataframe
        ncep_sfc_data = ncep_sfc_data[date.year] \
                            .sel(lon=row['longitude'], lat=row['latitude'], method='nearest') \
                            .to_dask_dataframe() \
                            .compute() \
                            .set_index(['time'])

        for level in self.levels:
            #features at different pressure level
            point = ncep_data.loc[level]
            p1w = point.rolling(7).mean()  # 1 Week mean
            p2w = point.rolling(14).mean() # 2 Week mean
            p3w = point.rolling(21).mean() # 3 Week mean
            # 
            v0w = point.loc[date]
            v1w = p1w.loc[date]
            v2w = p2w.loc[date]
            v3w = p3w.loc[date]
            #
            for data_var in self.ncep_data_vars:
                features["{0}_0w_lvl_{1}".format(data_var,level)] = v0w[data_var]
                features["{0}_1w_lvl_{1}".format(data_var,level)] = v1w[data_var]
                features["{0}_2w_lvl_{1}".format(data_var,level)] = v2w[data_var]
                features["{0}_3w_lvl_{1}".format(data_var,level)] = v3w[data_var]
        #features at surface level
        point = ncep_sfc_data
        p1w = point.rolling(7).mean()  # 1 Week mean
        p2w = point.rolling(14).mean() # 2 Week mean
        p3w = point.rolling(21).mean() # 3 Week mean
        # 
        v0w = point.loc[date]
        v1w = p1w.loc[date]
        v2w = p2w.loc[date]
        v3w = p3w.loc[date]
        #
        for data_var in self.ncep_sfc_data_vars:
            features["{0}_0w".format(data_var)] = v0w[data_var]
            features["{0}_1w".format(data_var)] = v1w[data_var]
            features["{0}_2w".format(data_var)] = v2w[data_var]
            features["{0}_3w".format(data_var)] = v3w[data_var] 

        return features
    
    def _get_day_info(self, df_features):
        """ """
        df_features['day'] = df_features.date.dt.day
        df_features['month'] = df_features.date.dt.month
        df_features['year']= df_features.date.dt.year
        df_features['week_day'] = df_features.date.dt.weekday
        df_features['weekofyear'] = df_features.date.dt.weekofyear
        df_features['is_winter'] = df_features.apply(lambda x: int(x['month'] in [12,1,2]), axis=1)
        df_features['is_autumn'] = df_features.apply(lambda x: int(x['month'] in [9,10,11]), axis=1)
        df_features['is_summer'] = df_features.apply(lambda x: int(x['month'] in [6,7,8]), axis=1)
        df_features['is_spring'] = df_features.apply(lambda x: int(x['month'] in [3,4,5]), axis=1)
        return df_features 
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        #to reduce the processing time
        file_path = DATA_PATH + 'env_features.pickle'
        with open(file_path, 'rb') as f:
            env_features = pickle.load(f)
        
        df_env_features = {}
        for row_id, row in tqdm(X.iterrows(), total=X.shape[0], desc="Environmental Features"):
            if row_id in env_features:
                df_env_features[row_id] = env_features[row_id].copy()
            else:
                df_env_features[row_id] = self._extract_features(row)
        df_env_features = self._get_day_info(pd.DataFrame(df_env_features.values()))  
        
        return df_env_features
    
    
class GeographicalFeaturesGenerator:
    """FeaturesGenerator to generate geographical features.
    
    It generates the features like land_type, district, federal_subject of fire incident area and 
    distance from forest, city. Also, number of forest/field/cities within the radius.
    
    Russian-cities Data Source:
    https://github.com/pensnarik/russian-cities
    
    """
    
    def __init__(self, forest_data, field_data, land_data, nature_forest_data, cities_data):
        self.forest_data = forest_data
        self.field_data = field_data 
        self.land_data = land_data 
        self.nature_forest_data = nature_forest_data
        self.cities_data = cities_data
    
    def _predict_label(self, df_train, df_test, label=None):    
        """predict the label(land_type,district..) for data points in df_test"""
        #train k-nearest neighbors classifier 
        neigh = KNeighborsClassifier(n_neighbors=5)
        X, y = df_train[['longitude', 'latitude']], df_train[label]
        neigh.fit(X, y)
        #predict the label for wildfire incidents
        pred_label = neigh.predict(df_test[['longitude', 'latitude']])
        return pred_label
    
    def _get_dst(self, df_train, df_test):
        """find the minimum distance from data points in df_test to nearest data point in df_train"""
        #train NearestNeighbors(Unsupervised learner)
        neigh = NearestNeighbors(1)
        neigh.fit(df_train[['longitude', 'latitude']])
        #find the K-neighbors of points in df_test
        distances, indices = neigh.kneighbors(df_test[['longitude', 'latitude']])
        return distances
    
    def _get_info_radius(self, df_train, df_test, radii, _type):
        """get number of forest/field/cities within the radius"""
        result = pd.DataFrame()
        #train NearestNeighbors(Unsupervised learner)
        neigh = NearestNeighbors()
        neigh.fit(df_train[['longitude', 'latitude']])
        #find
        for radius in radii:
            distances, indices = neigh.radius_neighbors(df_test[['longitude', 'latitude']], radius=radius)
            count = np.vectorize(len)(distances)
            has_type = np.where(count > 0, 1, 0)
            result['has_{0}_radius_{1}'.format(_type,radius)] = has_type
            result['num_{0}_radius_{1}'.format(_type,radius)] = count
        return result
    
    def _get_event_count_lastyear(self, wildfire_data):
        """ """
        radii = [1.0, 1.5, 2, 2.5]
        wildfire_data['year'] = wildfire_data.date.dt.year
        wildfire_data['month'] = wildfire_data.date.dt.month
        start_year = wildfire_data.year.min()
        end_year = wildfire_data.year.max()
        result = pd.DataFrame()
        for radius in radii:
            temp = pd.Series(np.zeros((wildfire_data.shape[0])))
            for cur_year, month in itertools.product(range(start_year+1,end_year+1),range(1,13)):
                prev_year = cur_year - 1
                mask_prev = (wildfire_data.year<=prev_year)&(wildfire_data.month==month)
                mask_cur = (wildfire_data.year==cur_year)&(wildfire_data.month==month)
                if sum(mask_prev)!=0 and sum(mask_cur)!=0:
                    #train
                    neigh = NearestNeighbors(radius=radius)
                    neigh.fit(wildfire_data[mask_prev][['longitude', 'latitude']])
                    #find
                    distances, indices = neigh.radius_neighbors(wildfire_data[mask_cur][['longitude', 'latitude']])
                    count = np.vectorize(len)(distances)
                    #
                    temp.loc[mask_cur] = count
            result['num_event_lastyear_radius_{0}'.format(radius)] = temp 
        return result
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print("Geographical Features........",end='   ')
        start = time.time()
        #predict land_type, federal_subject, district for fire incident area
        X['land_type'] = self._predict_label(self.land_data, X, label='land_type')
        X['federal_subject'] =  self._predict_label(self.cities_data, X, label='subject')
        X['district'] =  self._predict_label(self.cities_data, X, label='district')
        X['population'] =  self._predict_label(self.cities_data, X, label='population')
        #Get minimum distance from fire incident to nature forest, forest, field, city
        X['nature_forest_dst'] = self._get_dst(self.nature_forest_data, X)
        X['forest_dst'] = self._get_dst(self.forest_data, X)
        X['field_dst'] = self._get_dst(self.field_data, X)
        X['city_dst'] = self._get_dst(self.cities_data, X)
        #
        X = pd.concat([X, self._get_info_radius(self.forest_data, X, [0.2, 0.5, 1.0], 'forest')], axis=1)
        #
        X = pd.concat([X, self._get_info_radius(self.field_data, X, [0.2, 0.5, 1.0], 'field')], axis=1)
        #
        X = pd.concat([X, self._get_info_radius(self.cities_data, X, [5, 10, 15], 'cities')], axis=1)
        #
        X = pd.concat([X, self._get_event_count_lastyear(X)], axis=1)
        end = time.time()
        print("Total time taken", end-start, " sec")
        return X