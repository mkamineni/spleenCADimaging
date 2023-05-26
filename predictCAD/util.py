import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm

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