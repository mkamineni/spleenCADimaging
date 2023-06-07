import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm
import step_reg
from sklearn.impute import SimpleImputer
import datetime


def add_mri_times(data):
    '''
    Adding MRI acquisition times and age at MRI
    '''
    # opening up datafile with information about MRI acquisiton times and age at MRI and modifying columns to prepare for merge with main dataframe
    mri_data = pd.read_csv("gs://ukbb_spleen/MRI_acquisition_times.csv", sep=',')
    mri_data = mri_data.rename({'Unnamed: 0': 'ID', 'sex': 'Sex', 'enroll_age':'age'}, axis='columns')

    # calculate time to mri acquisition in days
    mri_data.acquisition_time = pd.DatetimeIndex(mri_data.acquisition_time).normalize()
    mri_data.enroll_date = pd.DatetimeIndex(mri_data.enroll_date).normalize()

    mri_data['time_to_mri_acquisition'] = (mri_data['acquisition_time']  - mri_data['enroll_date'] ).astype('timedelta64[D]').astype('int64')

    mri_data['age_at_mri'] = mri_data['age'] + (mri_data['acquisition_time']  - mri_data['enroll_date']).astype('timedelta64[D]').astype('int64')/365.25
    mri_data = mri_data[['ID', 'Sex', 'age', 'time_to_mri_acquisition', 'age_at_mri']]
    mri_data['age'] = mri_data['age'].apply(np.floor).astype(int)
    
    # merge mri data with cohort
    merged = data.merge(mri_data, on = ['ID', 'age', 'Sex'], how = 'inner')
    
    # update the age to age at mri
    new = merged.drop('age', axis = 1)
    new = new.rename(columns = {'age_at_mri': 'age'})
    return new
    

def add_radiomics_features(data, phase, organ = 'spleen'):
    print(organ)
    rad = pd.read_csv('../radiomics/radiomics_%s.csv' %organ)
    diag_cols = [elem for elem in rad.columns if 'diagnostics' in elem]
    rad = rad[rad.Image.str.contains(phase)]
    rad = rad.drop(['Image', 'Mask']+diag_cols, axis = 1)
    rad = rad.add_prefix(organ+'_')
    rad = rad.rename(columns = {organ+'_ID':'ID'})
    rad = rad.drop_duplicates('ID').dropna()
    covars = list(rad.columns)
    covars.remove("ID")

    print(rad.shape)
    data = data.drop_duplicates('ID')
    print(data.shape)
    data = pd.merge(data, rad, on = "ID", how = "inner")
    print(data.shape)
    for col in data.columns:
        if 'Unnamed' in col:
            data = data.drop(col, axis =1)
            covars.remove(col)

    return data, covars

def add_existing_abdominal_features(data_filt, abdominal_covars):
    exist_ab=pd.read_csv("gs://ukbb_spleen/Abdominal_features/abdominal_features.csv")
    abdominal_feat_map = {'f.eid': 'ID', 'f.21083.2.0':'spleen_vol'}
    exist_ab=exist_ab.rename(columns = abdominal_feat_map)
    data = pd.merge(data_filt, exist_ab, on = "ID", how = "inner")

    print(data.spleen_vol.value_counts(dropna=False))
    print(data.head())
    return data

def make_feat_numerical(coh, covars):
    
    cat_feats = ['race', 'Sex', 'SmokingStatusv2']
    feat_map = {'race':'race', 'Sex':'sex', 'SmokingStatusv2':'smoking'}
    # categories that are already binary: dm2_prev, dm1_prev, antihtnbase, so leave alone

    # continuous vars: tchol, hdl, SBP

    # quintile age
    #age_expand = pd.get_dummies(pd.qcut(coh['age'], 5), prefix = "age")
	#coh = pd.concat([coh, age_expand], axis=1)
        
    # make categorical vars into binary
    for feat in cat_feats:
        if feat in coh.columns:
            feat_expand = pd.get_dummies(coh[feat], prefix = feat_map[feat])
            coh = pd.concat([coh, feat_expand], axis=1)
            covars.extend(feat_expand.columns)
    
    coh = coh.drop([feat for feat in cat_feats if feat in coh.columns], axis = 1)
    covars = [var for var in covars if var not in cat_feats]
    return coh, covars

def impute_select_features_cox(X, Y, include):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(X)
    X = pd.DataFrame(imputer.transform(X), columns = X.columns).reset_index(drop = True)
    Y = Y.reset_index(drop = True)
    if type(include) == list:
        return X[include], Y
    included_feats = step_reg.forward_regression(X.drop(include, axis = 1, inplace = False), Y, threshold_in = 0.2) + [include]
    X = X[included_feats]
    vif = calculate_vif(X, included_feats)
    print(vif.to_string())

    return X, Y

    
def impute_select_features(X_train, X_test, Y_train, Y_test):
    '''
    include is a string that represents a feature to include in the model
    '''
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(X_train)
    X_train = pd.DataFrame(imputer.transform(X_train), columns = X_train.columns).reset_index(drop = True)
    Y_train = Y_train.reset_index(drop = True)
    X_test = pd.DataFrame(imputer.transform(X_test), columns = X_test.columns).reset_index(drop = True)
    
    included_feats = step_reg.forward_regression(X_train, Y_train, threshold_in = 0.1)

    X_train, X_test = X_train[included_feats], X_test[included_feats]
    vif = calculate_vif(X_train, included_feats)
    print(vif.to_string())

    return X_train, X_test, Y_train, Y_test

    
def calculate_vif(df, features):    
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]        
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)                
        
        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})

def make_filename(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa):
    filename = 'cohorts/CADcohort'
    if not dropNa:
        filename = filename + '_all'
    if withDemo:
        filename = filename + '_demo'
    elif not withPCE:
        filename = filename + '_just'
    if withRadiomicsSpleen:
        filename = filename + '_rad'
    if withRadiomicsLiver:
        filename = filename + '_radliver'    
    if withExistAbFeats:
        filename = filename + '_existab'
    return filename+'.csv'