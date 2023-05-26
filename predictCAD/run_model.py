import random
import numpy as np
random.seed(0)
np.random.seed(0)

# modules for Log Reg
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import xgboost as xgb
import step_reg

from logreg_wpval import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from util import calculate_vif, make_filename
    
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

            # configure the cross-validation procedure
            cv = KFold(n_splits=5, shuffle=True, random_state=1)

            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            imputer.fit(X_train)
            X_train = pd.DataFrame(imputer.transform(X_train), columns = X_train.columns).reset_index(drop = True)
            Y_train = Y_train.reset_index(drop = True)
            included_feats = step_reg.forward_regression(X_train, Y_train)
            X_train = X_train[included_feats]
            vif = calculate_vif(X_train, included_feats)
            f.write(vif.to_string() + "\n")

            # define search
            if 'logreg' in choice:
                # define search space
                c_parameters=[5**(-5),5**(-4),5**(-3),5**(-2),5**(-1),0.5,5**(0),5,10]

                if choice == 'logreg_l1':
                    pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression(solver = 'saga'))])
                    param_grid = dict({'model__penalty': ['l1'], 'model__max_iter': [1000, 5000], 'model__class_weight':['balanced', None], 'model__C':c_parameters})
                else:
                    pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression())])
                    param_grid = dict({'model__penalty': ['l2'], 'model__class_weight':['balanced', None], 'model__C':c_parameters})

            else:
                '''
                pipeline = Pipeline([
                    ('standard_scaler', StandardScaler()), 
                    ('pca', PCA()), 
                    ('model', xgb.XGBClassifier())
                ])
                '''
                # no need to scale before using xgboost because decision trees are insensitive to scaling!
                
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
            
            # execute search
            result = search.fit(np.array(X_train), Y_train)
            print(result)
            best_pipeline = result.best_estimator_
            X_test = pd.DataFrame(imputer.transform(X_test), columns = X_test.columns).reset_index(drop = True)
            X_test = X_test[included_feats]
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
                print(len(included_feats))
                tups = []
                for ind, coef in enumerate(feat_imp):
                    tups.append((coef, se[ind], p_values[ind]))

                print(feat_imp)
                feat_to_imp = dict(zip(included_feats, tups))

                sorted_feat_to_imp = sorted(feat_to_imp.items(), key=lambda x:x[1][2], reverse = False)
                print(sorted_feat_to_imp)

                f.write("Top 20 Features: %s \n" %str(sorted_feat_to_imp[:20])) 