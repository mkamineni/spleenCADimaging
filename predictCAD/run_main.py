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

    parser.add_argument('--eval_model', '-eval_model', 
    	help = "evaluate the performance of the model and calculate confidence intervals for coefficients",
    	action = 'store_true', 
    	default = False)
    
    parser.add_argument('--coh', '-coh', 
    	help = "cohort parameters, mention any or all of following: pce, demo, spleen, liver, existab, dropna",
    	default = False, 
    	type = str)
    



    args = parser.parse_args()
    randseed = args.randseed
    preprocess = args.preprocess
    run_bootstrap = args.run_bootstrap
    eval_model = args.eval_model
    coh = args.coh.lower()
    run_model = args.run_model
    model_choices = [elem.strip() for elem in args.model_choices.lower().split(',')]
    
    withPCE, withDemo, withRadiomicsSpleen = 'pce' in coh, 'demo' in coh, 'spleen' in coh
    withRadiomicsLiver, withExistAbFeats, dropNa = 'liver' in coh, 'existab' in coh, 'dropna' in coh
    
    if preprocess:
        create_cohort(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa)
    
    if run_model:
        train_model(model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa)
    
    if run_bootstrap:
        bootstrap_model(model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa)
    
    if eval_model:
        pass
