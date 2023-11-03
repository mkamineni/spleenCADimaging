import random
import numpy as np
random.seed(0)
np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import xgboost as xgb

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
    c_parameters=[5**(-5),5**(-4),5**(-3),5**(-2),5**(-1),0.5,5**(0),5,10]

    if model_choice == 'logreg_l1':
        pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression(solver = 'saga'))])
        param_grid = dict({'model__penalty': ['l1'], 'model__max_iter': [1000, 5000], 'model__class_weight':['balanced', None], 'model__C':c_parameters})
        
    elif model_choice == 'logreg_l2':
        pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression())])
        param_grid = dict({'model__penalty': ['l2'], 'model__class_weight':['balanced', None], 'model__C':c_parameters})

    elif model_choice == 'xgboost':                
        pipeline = xgb.XGBClassifier()

        param_grid = dict({
            'max_depth': [3, 5, 10, 15, 20],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            #'subsample': np.arange(0.5, 1.0, 0.1),
            #'colsample_bytree': np.arange(0.4, 1.0, 0.1),
            #'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
            'n_estimators': [100, 500, 1000]
        })        

    # configure the cross-validation procedure
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', n_jobs=-1, cv=cv)
    return search

def train_eval_cross_sectional_model(choice, X_train, Y_train, X_test, Y_test, cohort, outcome, threshold):
    search = initialize_model_grid_search(choice)
    result = search.fit(np.array(X_train), Y_train)
    print(result)
    predictions = result.best_estimator_.predict_proba(np.array(X_test))
    return predictions, result

def train_eval_cox_model(f, filename, data_filt, covars, outcome, threshold):
    #define the time outcome for patients who havee CAD
    time_outcome = 'Years_To_'+outcome.replace('Incident_', '')
    
    # drop patients with prevalent CAD
    print("Before dropping prevalent CAD cases %s" %str(data_filt.shape))
    data_filt = data_filt[(data_filt[time_outcome]>0)|(data_filt[time_outcome].isnull())]
    print("After dropping prevalent CAD cases %s" %str(data_filt.shape))
    
    # define time outcome for patients without CAD as time to last follow up
    data_filt[time_outcome] = np.where(data_filt[time_outcome].isnull(), data_filt['time_to_follow_up'], data_filt[time_outcome])
    
    # split the data into train and test sets (70% training)
    data_filt = data_filt.sample(frac=1.0)
    train = data_filt[:int(data_filt.shape[0]*0.7)]
    test = data_filt[int(data_filt.shape[0]*0.7):]
    
    Y_tr, Y_test= train[outcome], test[outcome]
    X_tr, Y_tr, X_test, Y_test = impute_select_features_cox(train[covars+[time_outcome]], Y_tr, \
                                      test[covars+[time_outcome]], Y_test, \
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
    test_score = cph.score(pd.concat([X_test, Y_test], axis=1))
    return test_score


def train_model(model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, outcomes):
    '''
    Train one iteration of the model, compatible with any of the model choices: logistic regression, xgboost, and cox model
    '''
    with open('cohorts/covars.txt', 'r') as fp:
        covars = [x.strip() for x in fp.readlines()]
    
    if withRadiomicsLiver:
        f = open('liver_model_outputs.txt', "a+")
    else:
        f = open('model_outputs.txt', "a+")
    f.write('-----------------Date of Analysis: %s----------------- \n' %str(date.today()))
    cohort = make_filename(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa)
    print(cohort)
    f.write(cohort + '\n')
    data_filt = pd.read_csv(cohort)
    train, test = data_filt[data_filt.train==1], data_filt[data_filt.train==0]
    X_train, X_test = train[covars], test[covars]
    f.write("Number of patients %d \n" %data_filt.shape[0])

    for choice in model_choices:
        f.write('Model Choice: %s \n' %choice)
        for outcome in outcomes:
            Y_train, Y_test = train[outcome], test[outcome]
            f.write("Outcome that is being predicted: %s \n" %outcome)
            filename = cohort[8:-4]+'_'+outcome #removing .csv at end of cohort name and cohorts directory at beginning
            
            thresholds = [0.025, 0.05, 0.1, 0.2]
            metrics = []
            for threshold in thresholds:
                if choice == 'cox':
                    # only use for 70% of the data for training and validation, test set is untouched
                    metric = train_eval_cox_model(None, filename, data_filt.sample(frac=0.7), covars, outcome, threshold)
                else:
                    X_train_tr, X_train_val, Y_train_tr, Y_train_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
                    X_train_tr, X_train_val, Y_train_tr, Y_train_val = impute_select_features(X_train_tr, X_train_val, \
                                                                                              Y_train_tr, Y_train_val, threshold)

                    predictions, result = train_eval_cross_sectional_model(choice, X_train_tr, Y_train_tr, 
                                                                           X_train_val, Y_train_val, cohort, outcome, threshold)
                    metric = standard_metrics(predictions, X_train_val, Y_train_val, log = None, model = choice, filename = filename)
                metrics.append(metric)
            print(metrics)
            best_threshold = thresholds[np.argmax(metrics)]
            
            # train final model after choosing best threshold
            if choice == 'cox':
                CI = train_eval_cox_model(f, filename, data_filt, covars, outcome, threshold)
            else:
                X_train, X_test, Y_train, Y_test = impute_select_features(X_train, X_test, Y_train, Y_test, best_threshold)
                predictions, result = train_eval_cross_sectional_model(choice, X_train, Y_train, \
                                                                       X_test, Y_test, cohort, outcome, best_threshold)
                standard_metrics(predictions, X_test, Y_test, log = f, model = choice, filename = filename)
                if 'logreg' in choice:
                    most_important_coefs(result, X_train, log = f)

            
def bootstrap_model(model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa, outcomes, reps = 1000):
    '''
    Function to implement Monte Carlo bootstrapping
    Only bootstraps if the model is logistic regression
    '''
    with open('cohorts/covars.txt', 'r') as fp:
        covars = [x.strip() for x in fp.readlines()]

    f = open('bootstrap_outputs.txt', "a+")
    f.write('-----------------Date of Analysis: %s----------------- \n' %str(date.today()))
    cohort = make_filename(withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa)
    print(cohort)
    f.write(cohort + '\n')
    data_filt = pd.read_csv(cohort)
    train, test = data_filt[data_filt.train==1], data_filt[data_filt.train==0]
    X_train, X_test = train[covars], test[covars]
    f.write("Number of patients %d \n" %data_filt.shape[0])
    
    for choice in model_choices:
        if 'logreg' in choice:
            for outcome in outcomes:
                f.write("Bootstrap of %d reps for Model %s and Outcome %s \n" %(reps, choice, outcome))
                Y_train, Y_test = train[outcome], test[outcome]

                X_train, X_test, Y_train, Y_test = impute_select_features(X_train, X_test, Y_train, Y_test)
                aucs = []
                coefs = np.zeros((reps, X_train.shape[1]))

                sample_size = X_train.shape[0]
                idx = [i for i in range(sample_size)]

                for i in range(reps): 
                    f1 = open('bootstrap_track.txt', 'a+')
                    f1.write("Starting bootstrap %d" %i)
                    print("Starting bootstrap %d" %i)
                    sidx = np.random.choice(idx,replace=True,size=sample_size)

                    X_train_b = X_train.to_numpy()[sidx,:]
                    Y_train_b = Y_train.to_numpy()[sidx]
                    predictions, result = train_eval_cross_sectional_model(f, choice, X_train_b, Y_train_b, X_test, Y_test, cohort, outcome)
                    fpr,tpr,_=metrics.roc_curve(np.array(Y_test), predictions[:,1], pos_label=1)
                    auroc=metrics.auc(fpr,tpr)
                    print('AUC')
                    print(auroc)
                    f1.write(str(auroc))
                    aucs.append(auroc)

                    best_pipeline = result.best_estimator_
                    best_model= best_pipeline.named_steps['model'] 
                    feat_imp = best_model.coef_[0]
                    print(np.array(feat_imp))
                    f1.write(str(np.array(feat_imp)))
                    f1.close()
                    coefs[i,:] = feat_imp
                bootstrap_eval(aucs, coefs, X_train.columns, reps, log = f)         
            
            

                
            
