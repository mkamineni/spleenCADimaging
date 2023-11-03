import pandas as pd
import numpy as np
import random
random.seed(5)
np.random.seed(5)

from util import add_all_mri_times, add_radiomics_features, add_existing_abdominal_features, calculate_vif, make_filename, make_feat_numerical

withPCE = False
withDemo = True
withRadiomicsLiver = True
withRadiomicsSpleen = False
withExistAbFeats = False #existing abdominal features
dropNa = False
phase = 'wat'

def create_cohort(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, outcomes, phase = 'wat'):

    data = pd.read_csv("gs://ukbb_spleen/CHIPCAD_pheno_v5.csv", sep='\s+')
    data_filt = data.sort_values(by = ["ID"]).drop_duplicates()
    print(data_filt.shape)

    pce_covars = ['age', 'race', 'Sex', 'tchol', 'hdl', 'SBP', 'dm2_prev', 'dm1_prev', 'antihtnbase', 'SmokingStatusv2', 'statin0']
    demo_only = ['age', 'race', 'Sex']
    other= ['ID', 'pce_goff','time_to_follow_up'] +['Years_To_'+outcome.replace('Incident_', '').replace('Prevalent_', '') for outcome in outcomes] # adding follow up times here because don't want to drop na based on these columns, unless cox model
    abdominal_covars = ['spleen_vol']
    
    data_filt = add_all_mri_times(data_filt)
    
    #if withRadiomicsSpleen:
    data_filt, spleen_rad_covars = add_radiomics_features(data_filt, phase, 'spleen')

    #if withRadiomicsLiver:
    data_filt, liver_rad_covars = add_radiomics_features(data_filt, phase, 'liver')    

    data_filt = add_existing_abdominal_features(data_filt, abdominal_covars)
    
    covars = []
    if withPCE:
        covars = covars + pce_covars
    
    if withDemo:
        covars = covars + demo_only

    if withRadiomicsSpleen:
        covars = covars + spleen_rad_covars
        
    if withRadiomicsLiver:
        covars = covars + liver_rad_covars
        
    if withExistAbFeats:
        covars = covars + abdominal_covars

    cols_to_keep = covars+outcomes+other

    data_filt = data_filt[cols_to_keep]

    coh, covars = make_feat_numerical(data_filt, covars) 
    

    print(coh.shape)
    if dropNa:
        coh = coh.dropna()
    else:
        coh = coh.dropna(subset = outcomes) 
    print(coh.shape)

    '''
    Printing Statistics about the Patient Cohort
    '''
    for outcome in outcomes:
        print("Stats for outcome %s" %outcome)
        print(coh[outcome].value_counts(dropna=False))
    print("Mean Age (SD): %f (%f)" %(np.mean(np.array(coh.age)), np.std(np.array(coh.age))))
    print("Number of Females in Cohort: %d" %(len(coh.sex_Male) - np.sum(coh.sex_Male)))

    coh['train'] = np.random.choice(2, coh.shape[0], p=[0.3, 0.7])

    with open('cohorts/covars.txt', 'w') as fp:
        fp.writelines(var + '\n' for var in covars)

    coh.to_csv(make_filename(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa), index = None)