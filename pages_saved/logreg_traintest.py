# --------------------------------------------------------------------------------------------------------------------------------------------
# Package imports
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split,RepeatedKFold,cross_val_score,cross_validate

import multiprocessing
nproc = max([1,multiprocessing.cpu_count()-2])
import warnings
warnings.filterwarnings("ignore")
randomstate = 42

import streamlit as st

# user functions
import sigman_logreg.Logistic_Regression as fsc
from sigman_logreg.logreg_stats import calc_mcfad, calc_mcfadden_R2, precision_recall_f1_score, test_accuracy_score, kfold_logreg
import logreg_inp as inp

# --------------------------------------------------------------------------------------------------------------------------------------------
# univariate logistic regression
# --------------------------------------------------------------------------------------------------------------------------------------------
"what is the session state",st.session_state

if st.session_state.import_complete == True:

    results_1_param = pd.DataFrame(columns=['Model', 'Accuracy', 'McFadden_R2', 'Param_name', 'Threshold_Value'])

    count = 0
    for i in range(len(X_labels)):
        term = X_labels[i]
        X_sel = X[:, i].reshape(-1,1)
        lr = LogisticRegression().fit(X_sel,y)
        acc = round(lr.score(X_sel,y), 2)
        mcfad_r2 = round(calc_mcfad(X_sel, y), 2)
        m, b = lr.coef_[0][0], lr.intercept_[0]
        row_i = {'Model': term, 'Accuracy': acc, 'McFadden_R2': mcfad_r2, 'Param_name': X_labelname_dict[term], 'Threshold_Value': -b/m}
        results_1_param = results_1_param.append(row_i, ignore_index=True)

    results_1_param = results_1_param.sort_values('McFadden_R2', ascending=False)
    results_1_param.head(10)

else:
    st.markdown("# :exlamation: Please complete imports first!")


# --------------------------------------------------------------------------------------------------------------------------------------------
# train/test split
# --------------------------------------------------------------------------------------------------------------------------------------------