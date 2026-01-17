import random
import numpy as np
random.seed(0)
np.random.seed(0)

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
#import xgboost as xgb

from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# files that I implemented
from logreg_wpval import LogisticRegression
from util import impute_select_features, impute_select_features_cox, calculate_vif, make_filename
from evaluation import standard_metrics, most_important_coefs, bootstrap_eval

def initialize_model_grid_search(model_choice):
    '''
    Grid search over various hyperparameters for logistic regression and xgboost model
    '''
    c_parameters=[5**(-5),5**(-4),5**(-3),5**(-2),5**(-1),0.5,5**(0),5,10]

    if model_choice == 'logreg_l1':
        pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression(solver = 'saga'))])
        param_grid = dict({'model__penalty': ['l1'], 'model__max_iter': [1000, 5000], 'model__class_weight':['balanced', None], 'model__C':c_parameters})
        
    elif model_choice == 'logreg_l2':
        pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression())])
        param_grid = dict({'model__penalty': ['l2'], 'model__class_weight':['balanced', None], 'model__C':c_parameters})

    #elif model_choice == 'xgboost':                
    #    pipeline = xgb.XGBClassifier()

    #    param_grid = dict({
    #        'max_depth': [3, 5, 10, 15, 20],
    #        'learning_rate': [0.01, 0.1, 0.2, 0.3],
    #        'n_estimators': [100, 500, 1000]
    #    })        

    # configure the cross-validation procedure
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', n_jobs=-1, cv=cv)
    return search

def train_eval_cross_sectional_model(choice, X_train, Y_train, X_test, Y_test, cohort, outcom):
    '''
    Perform grid search for logistic regression model and returns predictions and the summary of the model fitting
    '''
    search = initialize_model_grid_search(choice)
    result = search.fit(np.array(X_train), Y_train)
    print(result)
    predictions = result.best_estimator_.predict_proba(np.array(X_test))
    return predictions, result

def train_eval_cox_model(f, filename, X_train, Y_train, X_test, Y_test, time_outcome, covars, outcome, threshold):
    '''
    Impute and use forward regression to select features, fit a Cox regression model, and bootstrap on the test set to generate a 95% CI interval for CI. 
    '''
    X_tr, Y_tr, X_test, Y_test = impute_select_features_cox(X_train[covars+[time_outcome]], Y_train, \
                                      X_test[covars+[time_outcome]], Y_test, \
                                      time_outcome, threshold)
    
    # train Cox model on training data
    cph = CoxPHFitter(l1_ratio=1.0)
    cph.fit(pd.concat([X_tr, Y_tr], axis=1), duration_col = time_outcome, event_col = outcome)
    cph.print_summary()
    results = proportional_hazard_test(cph, pd.concat([X_tr, Y_tr], axis=1), time_transform='rank')
    results.print_summary()
    if f:
        f.write(cph.summary.to_string())
        f.write(results.summary.to_string())
    
    # evalute on testing data
    test_score = cph.score(pd.concat([X_test, Y_test], axis=1), scoring_method='concordance_index')
    
    # bootstrap on test set
    reps = 1000
    concordances = []

    sample_size = X_test.shape[0]
    idx = [i for i in range(sample_size)]

    for i in range(reps):
        sidx = np.random.choice(idx,replace=True,size=sample_size)
        X_test_b = X_test.loc[sidx,:]
        Y_test_b = Y_test.loc[sidx]
        test_score_b = cph.score(pd.concat([X_test_b, Y_test_b], axis=1), scoring_method='concordance_index')
        concordances.append(test_score_b)
        
    sorted_concords = np.sort(np.array(concordances))
    conf_interval = [sorted_concords[int(0.025 * reps)], sorted_concords[int(0.975 * reps)]]
    print("Finished bootstrapping")
    print(np.mean(sorted_concords))
    print(str(conf_interval))
    return test_score


def train_model(datapath, model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, removeHemeCancer, outcomes, withLivSens):
    '''
    Train one iteration of the model, compatible with logistic regression and cox model
    '''
    with open(datapath+'cohorts/covars.txt', 'r') as fp:
        covars = [x.strip() for x in fp.readlines()]
   	 
    if withRadiomicsLiver:
        f = open('liver_model_outputs.txt', "a+")
    else:
        f = open(datapath+'model_outputs.txt', "a+")
    f.write('-----------------Date of Analysis: %s----------------- \n' %str(date.today()))
    cohort = make_filename(datapath, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, removeHemeCancer, withLivSens)
    f.write(cohort + '\n')
    data_filt = pd.read_csv(cohort)
    f.write("Number of patients %d \n" %data_filt.shape[0])
   

    for choice in model_choices:
        f.write('Model Choice: %s \n' %choice)
        for outcome in outcomes:
            # specify training and testing sets
            if choice == 'cox':
                #define the time outcome for patients who have CAD
                time_outcome = 'Years_To_'+outcome.replace('Incident_', '')

                # drop patients with prevalent CAD
                print("Before dropping prevalent CAD cases %s" %str(data_filt.shape))
                data_filt = data_filt[(data_filt[time_outcome]>0)|(data_filt[time_outcome].isnull())]
                print("After dropping prevalent CAD cases %s" %str(data_filt.shape))

                # define time outcome for patients without CAD as time to last follow up
                data_filt[time_outcome] = np.where(data_filt[time_outcome].isnull(), data_filt['time_to_follow_up'], data_filt[time_outcome])
                data_filt = data_filt[data_filt[time_outcome]>=0]
                
                # split the data into train and test sets (70% training)
                data_filt = data_filt.sample(frac=1.0)
                train = data_filt[:int(data_filt.shape[0]*0.7)]
                test = data_filt[int(data_filt.shape[0]*0.7):]
                X_train, X_test = train[covars+[time_outcome]], test[covars+[time_outcome]]
            else:
                train, test = data_filt[data_filt.train==1], data_filt[data_filt.train==0]
                X_train, X_test = train[covars], test[covars]
                
            print(train.shape)
            input(test.shape)
                
                
            Y_train, Y_test = train[outcome], test[outcome]
            f.write("Outcome that is being predicted: %s \n" %outcome)
            filename = cohort[8:-4]+'_'+outcome #removing .csv at end of cohort name and cohorts directory at beginning
            
            
            '''
            Find the best threshold for feature selection
            '''
            thresholds = [0.025, 0.05, 0.1, 0.2]
            metrics_x = []
            X_train_tr, X_train_val, Y_train_tr, Y_train_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
            for threshold in thresholds:
                print(X_train.shape)
                print(Y_train.shape)
                if choice == 'cox':
                    metric = train_eval_cox_model(None, filename, X_train_tr, Y_train_tr, X_train_val, Y_train_val, time_outcome, covars, outcome, threshold)
                else:
                    X_train_tr, X_train_val, Y_train_tr, Y_train_val = impute_select_features(X_train_tr, X_train_val, \
                                                                                              Y_train_tr, Y_train_val, threshold)

                    predictions, result = train_eval_cross_sectional_model(choice, X_train_tr, Y_train_tr, 
                                                                           X_train_val, Y_train_val, cohort, outcome)
                    metric = standard_metrics(datapath, predictions, X_train_val, Y_train_val, log = None, model = choice, filename = filename)
                metrics_x.append(metric)
            best_threshold = thresholds[np.argmax(metrics_x)]
            f.write("Best threshold: %f \n" %best_threshold)
            
            '''
            Train final model after choosing best threshold
            '''
            if choice == 'cox':
                CI = train_eval_cox_model(f, filename, X_train, Y_train, X_test, Y_test, time_outcome, covars, outcome, best_threshold)
                f.write("Concordance on Test set: %f \n" %CI)
            else:
                X_train, X_test, Y_train, Y_test = impute_select_features(X_train, X_test, Y_train, Y_test, best_threshold)
                predictions, result = train_eval_cross_sectional_model(choice, X_train, Y_train, \
                                       X_test, Y_test, cohort, outcome)

                reps = 1000
                aucs = []

                print(X_test.shape)
                print(Y_test.shape)
                assert X_test.shape[0]==Y_test.shape[0]
                sample_size = X_test.shape[0]
                idx = [i for i in range(sample_size)]

                for i in range(reps):
                    sidx = np.random.choice(idx,replace=True,size=sample_size)
                    X_test_b = X_test.loc[sidx,:]
                    Y_test_b = Y_test.iloc[sidx]
                    predictions_b = result.best_estimator_.predict_proba(np.array(X_test_b))
                    df = pd.DataFrame()
                    df['prob'] = predictions_b[:,1]
                    df['label'] = np.array(Y_test_b)
                    fpr,tpr,_=metrics.roc_curve(df.label, df.prob, pos_label=1)
                    auroc=metrics.auc(fpr,tpr)

                    aucs.append(auroc)

                sorted_aucs = np.sort(np.array(aucs))
                conf_interval = [sorted_aucs[int(0.025 * reps)], sorted_aucs[int(0.975 * reps)]]
                print("Finished bootstrapping")
                print(np.mean(sorted_aucs))
                print(str(conf_interval))

                standard_metrics(datapath, predictions, X_test, Y_test, log = f, model = choice, filename = filename)
                if 'logreg' in choice:
                    most_important_coefs(result, X_train, log = f)

            

         
