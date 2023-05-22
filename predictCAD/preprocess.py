
import pandas as pd
import numpy as np

from feature_processing import add_radiomics_features, add_existing_abdominal_features, calculate_vif

withPCE = True
withRadiomics = True
withExistAbFeats = False #existing abdominal features
dropNa = True
phase = 'wat'

def make_feat_numerical(coh, covars):
    
    cat_feats = ['race.x', 'Sex.x', 'SmokingStatusv2.x']
    feat_map = {'race.x':'race.x', 'Sex.x':'sex.x', 'SmokingStatusv2.x':'smoking.x'}
    # categories that are already binary: dm2_prev, dm1_prev, antihtnbase, so leave alone

    # continuous vars: tchol, hdl, SBP

    # quintile age
    #age_expand = pd.get_dummies(pd.qcut(coh['age'], 5), prefix = "age")
	#coh = pd.concat([coh, age_expand], axis=1)
        
    # make categorical vars into binary
    for feat in cat_feats:
        feat_expand = pd.get_dummies(coh[feat], prefix = feat_map[feat])
        coh = pd.concat([coh, feat_expand], axis=1)
        covars.extend(feat_expand.columns)
    
    coh = coh.drop(cat_feats, axis = 1)
    covars = [var for var in covars if var not in cat_feats]
    #imputer = SimpleImputer(c)
    #for col in covars:
    #    coh[col] = imputer.fit_transform(coh[col].values.reshape(-1,1))[:,0]
    return coh, covars

    
def make_filename(withPCE, withRadiomics, withExistAbFeats, dropNa):
    filename = 'cohorts/CADcohort'
    if not dropNa:
        filename = filename + '_all'
    if not withPCE:
        filename = filename + '_just'
    if withRadiomics:
        filename = filename + '_rad'
    if withExistAbFeats:
        filename = filename + '_existab'
    return filename+'.csv'

data = pd.read_csv("gs://ukbb_spleen/CHIPCAD_pheno_v4.csv", sep='\s+')
data_filt = data.sort_values(by = ["ID"]).drop_duplicates()
print(data_filt.shape)
    
pce_covars = ['age.x', 'race.x', 'Sex.x', 'tchol.x', 'hdl.x', 'SBP.x', 'dm2_prev.x', 'dm1_prev.x', 'antihtnbase.x', 'SmokingStatusv2.x']
outcomes = ['Coronary_Artery_Disease', 'Coronary_Artery_Disease_INTERMEDIATE', 'Coronary_Artery_Disease_HARD', 'Coronary_Artery_Disease_SOFT']
other= ['ID', 'pce_goff.y']
abdominal_covars = ['spleen_vol']

data_filt, rad_covars = add_radiomics_features(data_filt, phase)
data_filt = add_existing_abdominal_features(data_filt, abdominal_covars)

if withRadiomics:
    pce_covars = pce_covars + rad_covars
    
if withExistAbFeats:
    pce_covars = pce_covars + abdominal_covars
    
cols_to_keep = pce_covars+outcomes+other

data_filt = data_filt[cols_to_keep]

coh, covars = make_feat_numerical(data_filt, pce_covars)

if not withPCE:
    coh, covars = data_filt[rad_covars+outcomes], rad_covars    

print(coh.shape)
if dropNa:
    coh = coh.dropna()
else:
    coh = coh.dropna(subset = outcomes) 
print(coh.shape)


np.random.seed(5)
# add train-test split
coh['train'] = np.random.choice(2, coh.shape[0], p=[0.3, 0.7])

with open('covars.txt', 'w') as fp:
    fp.writelines(var + '\n' for var in covars)
print(len(coh))

print(coh.columns)
print(coh.head())

vif = calculate_vif(coh, covars)
print(vif.to_string())
coh.to_csv(make_filename(withPCE, withRadiomics, withExistAbFeats, dropNa), index = None)