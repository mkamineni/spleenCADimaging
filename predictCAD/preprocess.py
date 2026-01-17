import pandas as pd
import numpy as np
import random
random.seed(5)
np.random.seed(5)

from util import add_all_mri_times, add_radiomics_features, add_existing_abdominal_features, calculate_vif, make_filename, make_feat_numerical

withPCE = False
withDemo = True
withRadiomicsLiver = False
withRadiomicsSpleen = True
withExistAbFeats = False #existing abdominal features
dropNa = False
phase = 'wat'

def create_cohort(datapath, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, outcomes, phase = phase, removeHemeCancer = False, withLivSens = False):

    pce_covars = ['age', 'race', 'Sex', 'tchol', 'hdl', 'SBP', 'dm2_prev', 'dm1_prev', 'antihtnbase', 'SmokingStatusv2', 'statin0']
    demo_only = ['age', 'race', 'Sex']
    other= ['ID', 'pce_goff', 'time_to_follow_up'] +['Years_To_'+outcome.replace('Incident_', '').replace('Prevalent_', '') for outcome in outcomes] # adding follow up times here because don't want to drop na based on these columns, unless cox model
    abdominal_covars = ['spleen_vol']
    liv_sens = ["prevalent_disease_Chronic_Liver_Disease_updated"]#, "prevalent_disease_Chronic_Liver_Disease"]


    # reading in patient characteristics
    data = pd.read_csv(datapath+"CHIPCAD_pheno_v5.csv", sep='\s+', usecols=pce_covars+['ID', 'pce_goff'])
    
    data_filt = data.sort_values(by = ["ID"]).drop_duplicates()
    print(data_filt.shape)

    # adding in MRI times and outcomes
    data_filt = add_all_mri_times(datapath, data_filt, outcomes)
    print(data_filt.columns)

    if withRadiomicsSpleen:
        data_filt, spleen_rad_covars = add_radiomics_features(datapath, data_filt, phase, 'spleen')
    print(data_filt.columns)

    if withRadiomicsLiver:
        data_filt, liver_rad_covars = add_radiomics_features(datapath, data_filt, phase, 'liver')    

    if withLivSens:
        liv_data = pd.read_csv("../../Data/liver_analysis/CAD_pat_char_with_liver.csv", usecols = ["ID"]+liv_sens)
        input(data_filt.shape)
        data_filt = pd.merge(data_filt, liv_data, on="ID", how="left")
        input(data_filt.shape)         
    #data_filt = add_existing_abdominal_features(data_filt, abdominal_covars)
    
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

    if withLivSens:
        covars = covars + liv_sens

    cols_to_keep = covars+outcomes+other

    data_filt = data_filt[cols_to_keep]
    print(data_filt.race.value_counts(dropna=False))

    coh, covars = make_feat_numerical(data_filt, covars) 
    

    # Comment this out if you don't care about specific outcomes
    print(coh.shape)
    if dropNa:
        coh = coh.dropna()
    else:
        coh = coh.dropna(subset = outcomes)
    print(coh.shape)
    print(coh.head(10))
    
    if removeHemeCancer:
        no_heme_coh = pd.read_csv(datapath+"radspleen_wo_heme_cancer.csv", usecols = ["ID"])
        coh = coh.merge(no_heme_coh, how = "inner", on = "ID")
        print("removed heme")
        print(coh.shape)
        print(coh.head(10))

    '''
    Printing Statistics about the Patient Cohort
    '''
    for outcome in outcomes:
        print("Stats for outcome %s" %outcome)
        print(coh[outcome].value_counts(dropna=False))
    print("Mean Age (SD): %f (%f)" %(np.mean(np.array(coh.age)), np.std(np.array(coh.age))))
    print("Number of Females in Cohort: %d" %(len(coh.sex_Male) - np.sum(coh.sex_Male)))
    
    '''
    Print statisitcs related to UK Biobank-annotated splenic volume
    '''
    if withExistAbFeats:
        print(coh['spleen_vol'].value_counts())
        print("Mean UKBiobank Spleen Vol (SD): %f (%f)" %(np.nanmean(np.array(coh['spleen_vol'])), np.nanstd(np.array(coh['spleen_vol']))))
        fem_coh = coh[coh.sex_Female == 1]
        print("Mean Female UKBiobank Spleen Vol (SD): %f (%f)" %(np.nanmean(np.array(fem_coh['spleen_vol'])), np.nanstd(np.array(fem_coh['spleen_vol']))))
        mal_coh = coh[coh.sex_Female == 0]
        print("Mean Male UKBiobank Spleen Vol (SD): %f (%f)" %(np.nanmean(np.array(mal_coh['spleen_vol'])), np.nanstd(np.array(mal_coh['spleen_vol']))))
        num_bins = 5
        mal_coh['age_bin'], groups = pd.qcut(mal_coh['age'], num_bins, labels=range(num_bins), retbins = True)
        print(groups)
        for group in range(num_bins):
            mal_coh_age = mal_coh[mal_coh.age_bin == group]
            print(group)
            print("Mean UKBiobank Spleen Vol for age bin (SD): %f (%f)" %(np.nanmean(np.array(mal_coh_age['spleen_vol'])), np.nanstd(np.array(mal_coh_age['spleen_vol']))))

    # training label for prevalent CAD analyses
    coh['train'] = np.random.choice(2, coh.shape[0], p=[0.3, 0.7])

    with open(datapath+'cohorts/covars.txt', 'w') as fp:
        fp.writelines(var + '\n' for var in covars)
    print(coh.time_to_follow_up.value_counts(dropna=False))

    coh.to_csv(make_filename(datapath, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, removeHemeCancer, withLivSens), index = None)
