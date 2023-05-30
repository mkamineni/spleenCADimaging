import random
import numpy as np
random.seed(0)
np.random.seed(0)

# modules for Log Reg
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import xgboost as xgb

from logreg_wpval import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from util import impute_select_features, calculate_vif, make_filename
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

    
def initialize_model_grid_search(model_choice):
    # params for logregs
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

    search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', n_jobs=-1, cv=cv)
    return search

def train_cross_sectional_model(f, choice, X_train, Y_train, X_test, Y_test):
    # configure the cross-validation procedure
    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    search = initialize_model_grid_search(choice)

    # execute search
    result = search.fit(np.array(X_train), Y_train)
    print(result)
    best_pipeline = result.best_estimator_
    predictions = best_pipeline.predict_proba(np.array(X_test))

    auc = roc_auc_score(np.array(Y_test), predictions[:,1])
    print(auc)
    f.write("AUC of model: %f \n \n" %auc)

    if 'logreg' in choice:
        print(result.best_params_)
        f.write("Best Parameters of Model: %s \n" %str(result.best_params_))

        best_model= best_pipeline.named_steps['model'] 
        feat_imp = best_model.coef_[0]
        print(feat_imp)

        p_values = best_model.p_values
        se = best_model.se

        print(len(p_values))
        print(len(X_train.columns))
        tups = []
        for ind, coef in enumerate(feat_imp):
            tups.append((coef, se[ind], p_values[ind]))

        print(feat_imp)
        feat_to_imp = dict(zip(X_train.columns, tups))

        sorted_feat_to_imp = sorted(feat_to_imp.items(), key=lambda x:x[1][2], reverse = False)
        print(sorted_feat_to_imp)

        f.write("Top 20 Features: %s \n" %str(sorted_feat_to_imp[:20])) 

def train_cox_model(f, data_filt, covars, outcome):
    #create a new train test split
    time_outcome = 'FollowUp_'+outcome
    cox_data_filt = data_filt[data_filt[time_outcome]>data_filt['time_to_mri_acquisition']/365.25]
    X_train, X_test, Y_train, Y_test = train_test_split(cox_data_filt[covars+[time_outcome]], cox_data_filt[outcome], test_size=0.3)
    X_train, X_test, Y_train, Y_test = impute_select_features(X_train, X_test, Y_train, Y_test, include = time_outcome)
    print(Y_train.head())
    # train Cox model
    cph = CoxPHFitter()
    cph.fit(pd.concat([X_train, Y_train], axis=1), duration_col = time_outcome, event_col = outcome)
    print(cph.print_summary())
    results = proportional_hazard_test(cph, pd.concat([X_train, Y_train], axis=1), time_transform='rank')
    print(results.print_summary(decimals=3, model="untransformed variables"))


def train_model(model_choices, withPCE, withDemo, withRadiomicsSpleen, withRadiomicsLiver, withExistAbFeats, dropNa):
    with open('cohorts/covars.txt', 'r') as fp:
        covars = [x.strip() for x in fp.readlines()]
    outcomes = ['Coronary_Artery_Disease', 'Coronary_Artery_Disease_INTERMEDIATE', 'Coronary_Artery_Disease_HARD', 'Coronary_Artery_Disease_SOFT']
    
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
            
            if choice == 'cox':
                # have to filter data for cox model and (X_train, X_test, Y_train, Y_test, include = time_outcome)
                train_cox_model(f, data_filt, covars, outcome)
            else:
                X_train, X_test, Y_train, Y_test = impute_select_features(X_train, X_test, Y_train, Y_test)
                train_cross_sectional_model(f, choice, X_train, Y_train, X_test, Y_test)

