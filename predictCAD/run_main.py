from argparse import ArgumentParser

from preprocess import create_cohort
from run_model import train_model, bootstrap_model

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument('--randseed', '-randseed', 
    	help = "random seed for subsampling", 
    	default = 11, 
    	type = int)

    parser.add_argument('--preprocess', '-preprocess', 
    	help = "run preprocessing script",
    	action = 'store_true', 
    	default = False)
    
    parser.add_argument('--run_model', '-run_model', 
    	help = "run model once",
    	action = 'store_true', 
    	default = False)

    parser.add_argument('--model_choices', '-model_choices', 
    	help = "specify models to train: mention any or all of the following separated by commas: logreg_l1, logreg_l2, xgboost, cox",
    	default = 'logreg_l1', 
    	type = str)

    parser.add_argument('--run_bootstrap', '-run_bootstrap', 
    	help = "bootstrap model 1000 times",
    	action = 'store_true', 
    	default = False)
    
    parser.add_argument('--coh', '-coh', 
    	help = "cohort parameters, mention any or all of following: pce, demo, spleen, liver, existab, dropna, prev, inc",
    	default = False, 
    	type = str)

    args = parser.parse_args()
    randseed = args.randseed
    preprocess = args.preprocess
    run_bootstrap = args.run_bootstrap
    coh = args.coh.lower()
    run_model = args.run_model
    model_choices = [elem.strip() for elem in args.model_choices.lower().split(',')]
    
    withPCE, withDemo, withRadiomicsSpleen = 'pce' in coh, 'demo' in coh, 'spleen' in coh
    withRadiomicsLiver, withExistAbFeats, dropNa = 'liver' in coh, 'existab' in coh, 'dropna' in coh
    
    # modify outcomes - default is all, but if you want prevalent or incident, will be changed here
    outcomes = ['Coronary_Artery_Disease', 'Coronary_Artery_Disease_INTERMEDIATE', 'Coronary_Artery_Disease_HARD', 'Coronary_Artery_Disease_SOFT','composite_mi_cad_stroke']

    outcomes = ['Coronary_Artery_Disease_INTERMEDIATE']

    if 'prev' in coh:
        outcomes = ['Prevalent_'+outcome for outcome in outcomes]
        
    if 'inc' in coh:
        outcomes = ['Incident_'+outcome for outcome in outcomes]
        
    if preprocess:
        create_cohort(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, outcomes)
    
    if run_model:
        train_model(model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, outcomes)
    
    if run_bootstrap:
        bootstrap_model(model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, outcomes)
