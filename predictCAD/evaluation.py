import random
import numpy as np
import os
random.seed(0)
np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats

def standard_metrics(datapath, predictions, X_test, Y_test, log, model, filename):
    figdir = datapath+'figures/'
    if not os.path.exists(figdir+filename+'/'):
        os.makedirs(figdir+filename+'/')
        
    # set up dataframe 
    df = pd.DataFrame()
    df['prob'] = predictions[:,1]
    df['label'] = np.array(Y_test)
    df['age'] = X_test.age
    
    # compute AUROC
    fpr,tpr,_=metrics.roc_curve(df.label, df.prob, pos_label=1)
    overall_auroc=metrics.auc(fpr,tpr)
    
    #Plot AUROC figure
    plt.figure()
    plt.plot(fpr,tpr,label='Test AUROC=%0.2f' %overall_auroc)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(figdir+filename+'/auroc.png')
    plt.clf()
    if log:
        log.write('AUC of model: %f \n' %overall_auroc)
    
    # compute AUPR
    pre,rec,_=metrics.precision_recall_curve(df.label, df.prob)
    aupr=metrics.average_precision_score(df.label, df.prob)

    # plot AUPR curve
    plt.figure()
    plt.plot(rec,pre,label='Test APR=%0.2f' %aupr)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(figdir+filename+'/aupr.png')
    plt.clf()

    # compute precision, recall, and F1 score at 95th percentile
    threshold=np.percentile(df.prob,95)
    df['pred'] = [1 if score > threshold else 0 for score in df.prob]
    precision = metrics.precision_score(df.label, df.pred)
    recall = metrics.recall_score(df.label, df.pred)
    f1_score = metrics.f1_score(df.label, df.pred)
    if log:
        log.write('Precision, Recall, F1 score at 95th Percentile: %f, %f, %f \n \n' %(precision, recall, f1_score))
    
    # compute calibration curve
    df['bin']=pd.qcut(df.prob,10) #bin by risk est (5 bins)
    df2=df.loc[:,['label','prob','bin']].groupby('bin',as_index=False).mean()
    
    plt.scatter(df2.prob,df2.label,s=80)
    plt.plot([0, 0.1], [0, 0.1], '--')
    plt.xlabel('Predicted Risk')
    plt.ylabel('Fraction of Positive Cases')
    plt.tight_layout()
    plt.savefig(figdir+filename+'/calibration.png',format='png',dpi=300)
    plt.clf()
    
    # visualize age distribution and stratify predictions by age
    num_bins = 5
    df['age_bin'], groups = pd.qcut(df.age, num_bins, labels=range(num_bins), retbins = True)
    print(groups)
    #input(df.age_bin.value_counts())
    plt.figure()
    for group in range(num_bins):
        df_age = df[df.age_bin == group]
        fpr,tpr,_=metrics.roc_curve(df_age.label, df_age.prob, pos_label=1)
        auroc=metrics.auc(fpr,tpr)

        #Plot
        plt.plot(fpr,tpr,label='AUROC for age group %s-%s =%0.2f' %(str(groups[group]),str(groups[group+1]), auroc))
        if log:
            log.write('AUC for age group %s: %f \n' %(str(groups[group]), auroc))
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(figdir+filename+'/auroc_age.png')
    plt.clf()

    # return auroc on test set
    return overall_auroc

    
def most_important_coefs(result, X_train, log):
    log.write('Best Parameters of Model: %s \n' %str(result.best_params_))

    best_pipeline = result.best_estimator_
    best_model= best_pipeline.named_steps['model'] 
    feat_imp = best_model.coef_[0]

    p_values = best_model.p_values
    se = best_model.se

    assert len(p_values) == len(X_train.columns)
    tups = []
    for ind, coef in enumerate(feat_imp):
        tups.append((coef, se[ind], p_values[ind]))

    feat_to_imp = dict(zip(X_train.columns, tups))

    sorted_feat_to_imp = sorted(feat_to_imp.items(), key=lambda x:x[1][2], reverse = False)

    log.write('Top 20 Features: %s \n' %str(sorted_feat_to_imp[:20])) 
    
    adjusted_p_values = calculate_FDR(p_values)
    log.write('Top 20 Features adjusted p_values: %s \n' %str(adjusted_p_values)) 
    

def bootstrap_eval(aucs, coefs, covars, reps, log):
    def se(data):
        return np.std(data, ddof=1) / np.sqrt(np.size(data))
    sorted_aucs = np.sort(np.array(aucs))
    conf_interval = [sorted_aucs[int(0.025 * reps)], sorted_aucs[int(0.975 * reps)]]
    log.write("AUC [CI]: %f %s \n" %(np.mean(sorted_aucs), str(conf_interval)))

    coef_tups = [(np.mean(coefs[:,ind]), se(coefs[:,ind])) for ind in range(coefs.shape[1])]
    coef_tups_w_pvals = []
    for ind, tup in enumerate(coef_tups):
        z_score = tup[0]/tup[1] # z-score for eaach model coefficient
        #print(z_score)
        p_value = stats.norm.sf(abs(z_score))*2 ### two tailed test for p-values
        coef_tups_w_pvals.append((tup[0], tup[1], p_value))

    feat_to_imp = dict(zip(covars, coef_tups_w_pvals))
    
    sorted_feat_to_imp = sorted(feat_to_imp.items(), key=lambda x:x[1][2], reverse = False)

    log.write('Top 20 Features: %s \n' %str(sorted_feat_to_imp[:20])) 

def calculate_FDR(p_values):
    p_values = np.array(p_values)
    # Number of tests
    m = len(p_values)
    # Sort the p-values and get the sorted indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    # Calculate the cumulative minimum of the adjusted p-values (from the end)
    adjusted_p_values = np.minimum.accumulate((m / np.arange(m, 0, -1)) * sorted_p_values[::-1])[::-1]
    # Adjust the p-values to ensure they are monotonically increasing
    adjusted_p_values = np.minimum.accumulate(np.minimum(adjusted_p_values[::-1], 1))[::-1]
    return adjusted_p_values


#cox_p_values = [3.070157*10**(-2), 8.972309*10**(-3), 1.986826*10**(-3), 9.215075*10**(-3), 4.041386*10**(-2)]
cox_p_values = [9.192007*10**(-6), 2.280038*10**(-4)]

print(calculate_FDR(cox_p_values))
