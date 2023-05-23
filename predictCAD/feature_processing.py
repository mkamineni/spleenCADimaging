import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm

def add_radiomics_features(data, phase):
    rad = pd.read_csv('../radiomics/radiomics_spleen.csv')
    diag_cols = [elem for elem in rad.columns if 'diagnostics' in elem]
    rad = rad[rad.Image.str.contains(phase)]
    rad = rad.drop(['Image', 'Mask']+diag_cols, axis = 1)
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


'''
#Not currently being used anywhere, but was tested
def logit_pvalue(model, x):
   
    p1 = model.predict_proba(x)
    n1 = len(p1)
    input(n1)
    m1 = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    answ = np.zeros((m1, m1))
    for i in range(n1):
        add = np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * float(p1[i,1]) * float(p1[i, 0])
        answ = answ + add
        print(add)
    vcov = np.linalg.inv(np.matrix(answ))
    se = np.sqrt(np.diag(vcov))
    t1 =  coefs/se  
    p1 = (1 - norm.cdf(abs(t1))) * 2
    return p1
'''
