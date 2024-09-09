import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm
import step_reg
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import datetime
import random
random.seed(0)
np.random.seed(0)


#double check the race breakdown
#think about the split for incident and prevalent CAD
#think about how to group clusters
#double check Cox part in train_eval_cox_model

def add_all_mri_times(data, outcomes):
    '''
    adding information about prevalent and incident cases based on MRI
    '''
    mri_data = pd.read_csv("gs://ukbb_spleen/CAD_Phenotypes_for_MRI_Participants.csv")
    #mri_data = pd.read_csv("gs://ukbb_spleen/CAD_Phenotypes_Imaging_Participants.csv")

    mri_data = mri_data.rename(columns = {'f.eid':'ID', 'Index_Date':'MRI_Date'})
    
    # only want one MRI for patient, default for drop duplicates is to take the first one
    mri_data = mri_data.sort_values(by = ['ID', 'MRI_Date'], ascending = True).drop_duplicates('ID')
    
    # remove the 127 patients that got MRI on 3rd visit because I don't have that info
    mri_data = mri_data[~mri_data['filename'].str.contains('_3_0.zip')]
    
    # limit to outcomes, follow up times, and years after MRI to CAD diagnosis
    mri_data = mri_data[['ID', 'time_to_follow_up']+outcomes+['Years_To_'+outcome.replace('Incident_', '').replace('Prevalent_', '') for outcome in outcomes]]

    # merge mri data with cohort
    merged = data.merge(mri_data, on = ['ID'], how = 'inner')
    return merged
    

def add_radiomics_features(data, phase, organ = 'spleen'):
    '''
    Combine raddiomics features with cohort
    '''
    print(organ)
    print(phase)
    rad = pd.read_csv('../radiomics/radiomics_%s_1.csv' %organ)
    diag_cols = [elem for elem in rad.columns if 'diagnostics' in elem]
    rad = rad[rad.Image.str.contains(phase)]
    rad = rad.drop(['Image', 'Mask']+diag_cols, axis = 1)
    rad = rad.add_prefix(organ+'_')
    rad = rad.rename(columns = {organ+'_ID':'ID'})
    rad = rad.drop_duplicates('ID').dropna()
    covars = list(rad.columns)
    covars.remove("ID")

    data = data.drop_duplicates('ID')
    data = pd.merge(data, rad, on = "ID", how = "inner")
    
    # remove unnamed columns
    for col in data.columns:
        if 'Unnamed' in col:
            data = data.drop(col, axis =1)
            covars.remove(col)

    return data, covars

def add_existing_abdominal_features(data_filt, abdominal_covars):
    '''
    Add UK Biobank annotated spleen volume
    '''
    exist_ab=pd.read_csv("gs://ukbb_spleen/Abdominal_features/abdominal_features.csv")
    abdominal_feat_map = {'f.eid': 'ID', 'f.21083.2.0':'spleen_vol'}
    exist_ab=exist_ab.rename(columns = abdominal_feat_map)
    data = pd.merge(data_filt, exist_ab, on = "ID", how = "inner")
    return data

def make_feat_numerical(coh, covars):
    '''
    Expand categorical variables into binary indicator variables
    '''
    cat_feats = ['race', 'Sex', 'SmokingStatusv2']
    feat_map = {'race':'race', 'Sex':'sex', 'SmokingStatusv2':'smoking'}
        
    # make categorical vars into binary
    for feat in cat_feats:
        if feat in coh.columns:
            feat_expand = pd.get_dummies(coh[feat], prefix = feat_map[feat])
            coh = pd.concat([coh, feat_expand], axis=1)
            covars.extend(feat_expand.columns)
    
    coh = coh.drop([feat for feat in cat_feats if feat in coh.columns], axis = 1)
    covars = [var for var in covars if var not in cat_feats]
    return coh, covars

def impute_select_features_cox(X_tr, Y_tr, X_test, Y_test, time_outcome, threshold):
    '''
    Impute selected features for cox regression analysis for incident CAD. 
    Imputing only required when using all pce covariates. 
    Time_outcome is the time to the outcome of interest in a Cox regression model and should always be included in the final dataset
    Also apply standard scaling here
    '''
    # impute features with median of column value
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(X_tr)
    
    # apply imputation to train and test data
    X_tr = pd.DataFrame(imputer.transform(X_tr), columns = X_tr.columns).reset_index(drop = True)
    Y_tr = Y_tr.reset_index(drop = True)
    X_test = pd.DataFrame(imputer.transform(X_test), columns = X_test.columns).reset_index(drop = True)
    Y_test = Y_test.reset_index(drop = True)
    
    # use forward regression to identify most significant features
    included_feats = step_reg.forward_regression(X_tr.drop(time_outcome, axis = 1, inplace = False), Y_tr, threshold_in = threshold) + [time_outcome]
    X_tr = X_tr[included_feats]
    X_test = X_test[included_feats]
    covars = [elem for elem in included_feats if elem!=time_outcome]
    print(included_feats)
    print("Number of features selected %d" %len(included_feats))
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr[covars].to_numpy())
    X_tr_scaled = pd.DataFrame(X_tr_scaled, columns=covars)
    X_tr_scaled[time_outcome] = X_tr[time_outcome]
    
    X_test_scaled = scaler.transform(X_test[covars].to_numpy())
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=covars)
    X_test_scaled[time_outcome] = X_test[time_outcome]

    # calculate the variance inflation factor for the selected features
    vif = calculate_vif(X_tr_scaled, included_feats)
    print(vif.to_string())

    return X_tr_scaled, Y_tr, X_test_scaled, Y_test

    
def impute_select_features(X_train, X_test, Y_train, Y_test, threshold):
    '''
    Impute features for prevalent CAD logistic regression analysis 
    '''
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(X_train)
    X_train = pd.DataFrame(imputer.transform(X_train), columns = X_train.columns).reset_index(drop = True)
    Y_train = Y_train.reset_index(drop = True)
    X_test = pd.DataFrame(imputer.transform(X_test), columns = X_test.columns).reset_index(drop = True)
    
    included_feats = step_reg.forward_regression(X_train, Y_train, threshold_in = threshold)

    X_train, X_test = X_train[included_feats], X_test[included_feats]
    vif = calculate_vif(X_train, included_feats)
    print(vif.to_string())

    return X_train, X_test, Y_train, Y_test

    
def calculate_vif(df, features):    
    '''
    Calculate VIF to understand collinearity between features
    '''
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

def make_filename(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, removeHemeCancer):
    '''
    Based on the parameters passed to make cohort, create an identiifable filename
    '''
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
    if removeHemeCancer:
        filename = filename + '_no_heme'        
    return filename+'.csv'



