
import pandas as pd
import numpy as np
import random
random.seed(5)
np.random.seed(5)

from feature_processing import add_radiomics_features, add_existing_abdominal_features, calculate_vif

withPCE = False
withDemo = True
withRadiomicsLiver = True
withRadiomicsSpleen = False
withExistAbFeats = False #existing abdominal features
dropNa = False
phase = 'wat'

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

def create_cohort(withPCE, withDemo, withRadiomicsLiver, withRadiomicsSpleen, withExistAbFeats, dropNa, phase = 'wat'):

    data = pd.read_csv("gs://ukbb_spleen/CHIPCAD_pheno_v5.csv", sep='\s+')
    data_filt = data.sort_values(by = ["ID"]).drop_duplicates()
    print(data_filt.shape)

    pce_covars = ['age', 'race', 'Sex', 'tchol', 'hdl', 'SBP', 'dm2_prev', 'dm1_prev', 'antihtnbase', 'SmokingStatusv2', 'statin0']
    demo_only = ['age', 'race', 'Sex']
    outcomes = ['Coronary_Artery_Disease', 'Coronary_Artery_Disease_INTERMEDIATE', 'Coronary_Artery_Disease_HARD', 'Coronary_Artery_Disease_SOFT']
    other= ['ID', 'pce_goff']
    abdominal_covars = ['spleen_vol']

    if withRadiomicsSpleen:
        data_filt, rad_covars = add_radiomics_features(data_filt, phase, 'spleen')
    if withRadiomicsLiver:
        data_filt, rad_covars = add_radiomics_features(data_filt, phase, 'liver')    

    data_filt = add_existing_abdominal_features(data_filt, abdominal_covars)

    if withDemo:
        pce_covars = demo_only

    if withRadiomicsSpleen or withRadiomicsLiver:
        pce_covars = pce_covars + rad_covars

    if withExistAbFeats:
        pce_covars = pce_covars + abdominal_covars

    cols_to_keep = pce_covars+outcomes+other

    data_filt = data_filt[cols_to_keep]

    coh, covars = make_feat_numerical(data_filt, pce_covars)

    if not withPCE and not withDemo:
        coh, covars = data_filt[rad_covars+outcomes], rad_covars    

    print(coh.shape)
    if dropNa:
        coh = coh.dropna()
    else:
        coh = coh.dropna(subset = outcomes) 
    print(coh.shape)

    coh['train'] = np.random.choice(2, coh.shape[0], p=[0.3, 0.7])

    with open('cohorts/covars.txt', 'w') as fp:
        fp.writelines(var + '\n' for var in covars)

    coh.to_csv(make_filename(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa), index = None)