# modules for Log Reg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import xgboost as xgb
import step_reg

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from feature_processing import calculate_vif, logit_pvalue
    
with open('cohorts/covars.txt', 'r') as fp:
    covars = [x.strip() for x in fp.readlines()]
outcomes = ['Coronary_Artery_Disease', 'Coronary_Artery_Disease_INTERMEDIATE', 'Coronary_Artery_Disease_HARD', 'Coronary_Artery_Disease_SOFT']

cohort = 'cohorts/CADcohort_all_rad'#'CADcohort_just_rad'

choices = ['logreg_l1']#, 'logreg_l2']#, 'xgboost']
f = open('model_outputs.txt', "a+")
f.write('-----------------Date of Analysis: %s----------------- \n' %str(date.today()))
f.write(cohort + '\n')
data_filt = pd.read_csv(cohort +'.csv')
train, test = data_filt[data_filt.train==1], data_filt[data_filt.train==0]
X_train, X_test = train[covars], test[covars]
f.write("Number of patients %d" %data_filt.shape[0])

for choice in choices:
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
        input(included_feats)
        X_train = X_train[included_feats]
        vif = calculate_vif(X_train, included_feats)
        input(vif.to_string())

        # define search
        if 'logreg' in choice:
            # define search space
            #c_parameters=[5**(-14),5**(-13),5**(-12),5**(-11),5**(-10),5**(-9),5**(-8),5**(-7),5**(-6),5**(-5),5**(-4),5**(-3),5**(-2),5**(-1),5**(0)]
            c_parameters=[5**(-5),5**(-4),5**(-3),5**(-2),5**(-1),0.5,5**(0),5,10]
            
            if choice == 'logreg_l1':
                pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression(solver = 'saga'))])
                param_grid = dict({'model__penalty': ['l1'], 'model__max_iter': [1000, 5000], 'model__class_weight':['balanced', None], 'model__C':c_parameters})
            else:
                pipeline=Pipeline([('standard_scaler', StandardScaler()), ('model', LogisticRegression())])
                param_grid = dict({'model__penalty': ['l2'], 'model__class_weight':['balanced', None], 'model__C':c_parameters})

        else:
            model = xgb.XGBClassifier()

            pipeline = Pipeline([
                ('standard_scaler', StandardScaler()), 
                ('pca', PCA()), 
                ('model', model)
            ])

            param_grid = {
                'pca__n_components': [5, 10, 15],        # will need to change this when adding more features
                'model__max_depth': [2, 3, 5, 7, 10],
                'model__n_estimators': [10, 100, 500],
            }

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', n_jobs=-1, cv=cv)
        # execute search
        result = search.fit(X_train, Y_train)
        print(result)
        best_model = result.best_estimator_
        X_test = imputer.transform(X_test)
        #X_test = X_test[included_feats]
        predictions = best_model.predict_proba(X_test)
        print(result.best_params_)
        f.write("Best Parameters of Model: %s \n" %str(result.best_params_))

        feat_imp = best_model.named_steps['model'].coef_[0]
        #p_values = logit_pvalue(best_model.named_steps['model'], X_train)

        #tups = []
        #for ind, coef in enumerate(feat_imp):
        #    tups.append((coef, p_values[ind]))
            
        print(feat_imp)
        print(covars)
        feat_to_imp = dict(zip(covars, feat_imp))

        sorted_feat_to_imp = sorted(feat_to_imp.items(), key=lambda x:x[1], reverse = True)
        print(sorted_feat_to_imp)
        
        f.write("Top 20 Features: %s \n" %str(sorted_feat_to_imp[:20])) 


        print(predictions[:,1])
        print(np.array(Y_test))
        auc = roc_auc_score(np.array(Y_test), predictions[:,1])
        print(auc)
        f.write("AUC of model: %f \n \n" %auc)
    