from argparse import ArgumentParser

from preprocess import create_cohort

if __name__=="__main__":
	parser = ArgumentParser()

	parser.add_argument('--randseed', '-randseed', 
		help = "random seed for subsampling", 
		default = 11, 
		type = int)

	parser.add_argument('--preprocess', '-preprocess', 
		help = "run preprocessing script",
		default = False, 
		type = str)

	parser.add_argument('--run_logreg', '-run_logreg', 
		help = "run logistic regression model with cohort",
		default = False, 
		type = str)

	parser.add_argument('--run_bootstrap', '-run_bootstrap', 
		help = "bootstrap model 1000 times",
		default = True, 
		type = str)

	parser.add_argument('--eval_model', '-eval_model', 
		help = "evaluate the performance of the model and calculate confidence intervals for coefficients",
		default = False, 
		type = str)
    
	parser.add_argument('--coh', '-coh', 
		help = "cohort parameters, mention any or all of following: pce, demo, spleen, liver, existab, dropna",
		default = False, 
		type = str)


	args = parser.parse_args()
	randseed = args.randseed
	preprocess = args.preprocess
	run_logreg = args.run_logreg
	run_bootstrap = args.run_bootstrap
	eval_model = args.eval_model
    coh = args.coh.lower()
    
    withPCE, withDemo, withRadiomicsLiver = 'pce' in coh, 'demo' in coh, 'liver' in coh
    withRadiomicsSpleen, withExistAbFeats, dropNa = 'spleen' in coh, 'existab' in coh, 'dropna' in coh

    if preprocess:
        create_cohort(withPCE, withDemo, withRadiomicsLiver, withRadiomicsSpleen, withExistAbFeats, dropNa)
    
    if run_logreg:
        pass
    
    if run_bootstrap:
        pass
    
    if eval_model:
        pass
