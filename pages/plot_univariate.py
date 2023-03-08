import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split,RepeatedKFold,cross_val_score,cross_validate

import sigman_logreg.Logistic_Regression as fsc
from sigman_logreg.logreg_stats import calc_mcfad, calc_mcfadden_R2, precision_recall_f1_score, test_accuracy_score, kfold_logreg 

import multiprocessing
nproc = max([1,multiprocessing.cpu_count()-2])
import warnings
warnings.filterwarnings("ignore")
randomstate = 42

import streamlit as st

# Plot top univariates
# Direct copy and paste from Sigman repo, adding in streamlit widgets when needed for user input

st.markdown("# Plot top univariate logistic regression models")

num_plots = 5  #Specify how many univariates to plot
skipfeatures = []  # add any features that you don't want to plot, e.g. ['x1', 'x3']


df_combined = pd.DataFrame(np.hstack((X,y[:,None]))) 
newcols = ["x"+str(i+1) for i in df_combined.columns.values]
df_combined.columns = newcols
response = newcols[-1]
df_combined.rename(columns={response:"y"},inplace=True)
df_combined.drop(skipfeatures,axis=1,inplace=True)

for i in range(num_plots):
    param_i = results_1_param.iloc[i].Model
    param_name = X_labelname_dict[param_i]
    threshold = results_1_param.iloc[i].Threshold_Value
    print("{} {}. Threshold value {:.2f}".format(param_i, param_name, threshold))
    print("Accuracy: {:.0f}%".format(100*results_1_param.iloc[i].Accuracy))
    print("McFadden R2: {}".format(results_1_param.iloc[i].McFadden_R2))
    plot_fit_1D(param_i, df_combined, save_fig=False)
    print("\n\n")

st.markdown("# Plot univariate logistic regression with user-defined parameter")
plot_params = st.multiselect("Which parameters should be used? Select all required.",compinp.columns)
# plot_params = ['x386','x240'] #populate with a list of parameters (e.g. plot_params = ['x1', 'x20', 'x21'])

for param_i in plot_params:
    row = results_1_param[results_1_param.Model == param_i]
    param_name = X_labelname_dict[param_i]
    threshold = float(row.Threshold_Value)
    acc = float(row.Accuracy)
    mcfad_r2 = float(row.McFadden_R2)
    print("{} {}. Threshold value {:.2f}".format(param_i, param_name, threshold))
    print("Accuracy: {:.0f}%".format(100*acc))
    print("McFadden R2: {}".format(mcfad_r2))
    plot_fit_1D(param_i,df_combined)
    print("\n\n")