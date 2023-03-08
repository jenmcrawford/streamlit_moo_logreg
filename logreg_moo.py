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

## Sidebar to set global settings
# init - use session states in the future
expinp = None
compinp = None

st.sidebar.markdown("## Import experimental data")
exp_up = st.sidebar.file_uploader("Results?",type=".xlsx")
if exp_up:
    exp_sheet = 'Data' # data excel sheet name
    exp_num_samples = 42 # include all the ligands, the empty rows will be removed later
    response_col = 2 # 0-indexed, the column containing the experimental results
    y_label_col_exp= 1 # 0-indexed, the number of the column with the ligand ids
    expinp = pd.read_excel(exp_up,exp_sheet,header=2,index_col=y_label_col_exp,nrows=exp_num_samples,usecols=list(range(0,response_col+1)),engine='openpyxl') #feels like there's a better way -- not going to bother improving

par_opt = st.sidebar.radio("Do you need to upload your own parameter set?",("No","Yes"))
if par_opt == "Yes":
    par_df = st.sidebar.file_uploader("Computed parameter set",type=".xlsx") #maybe need to process to be in same form as standard compinp
    compinp = pd.read_excel(par_df)
if par_opt == "No":
    compinp = pd.read_excel("Bisphosphine_Parameters.xlsx",sheet_name="symm_adapt_lowconf")

y_cut = st.sidebar.number_input("Determine threshold value for hit/no-hit:",min_value=0.0,max_value=100.0)

## Run through import
st.markdown("# :cow2: Multi-Objective Optimization :cow2:")

## Process imports

if compinp is not None:
    compinp = compinp.drop(['smiles'],axis=1)
    par_start_col = 1
    compinp.index = compinp.index.map(str)

    col1, col2 = st.columns(2)

    with col1:
        y_label_col_comp = st.selectbox("Which column contains ligand IDs?",options=compinp.columns)

    with col2:
        if expinp is not None:
            option = st.selectbox(
                "What is the reaction output?",
                options=expinp.columns
            )
            response_col = expinp.columns.get_loc(option)
    # names of the descriptors
    X_names = list(compinp.iloc[0,-1])

    # labels for descriptors e.g. x1, x2, x3... make list then take only selection for descriptors
    X_labels = list(compinp.columns)[par_start_col-1:-1]

    # removes names of the descriptors from compinp
    compinp.drop(index=compinp.index[0],inplace=True)

    # X_all = np array of descriptor values for all ligands
    X_all = np.asarray(compinp[X_labels],dtype=np.float)

    # np array of the ligand ids from the descriptor file
    y_labels_comp = np.asarray(list(compinp.index),dtype=str)

    # compnan = array of True/False designating if a ligand has a missing descriptor value(s) or not
    # nan = not a number. isnan returns True if Nan, in this case for any value in a row (axis of 1 = row).
    compnan = np.isnan(X_all).any(axis=1)

    # compares the arrays, and keeps the sample in y_labels_comp/X_all if the corresponding value in ~compnan = True.
    # ~ means it inverts True and False in compnan. This is removing any ligands missing descriptors.
    y_labels_comp,X_all = y_labels_comp[~compnan],X_all[~compnan]

    # combines the labels and names of descriptors as a single value in a list e.g. "x1 Vmin_LP(P).Boltz" 
    X_labelname = [" ".join(i) for i in zip(X_labels,X_names)]

    # makes a dictionary with key of descriptor label, value of descriptor name
    X_labelname_dict = dict(zip(X_labels,X_names))

    # heading ('label') for the response column
    resp_label = list(expinp.columns)[response_col-1]

    # array of the experimental results
    y = np.asarray(expinp.iloc[:,response_col-1],dtype=np.float)

    # array of all the ligand ids in the experimental file (curated below to give ligands with results only)
    y_labels_exp = np.asarray(list(expinp.index),dtype=str)

    # array with True for experimental results present, False for none
    mask_y = ~np.isnan(y)

    # check if each value of y_labels_exp (ligand ids in exp file) is also in y_labels_comp (ligand ids in descriptor file),
    # to give an array with True/False if a match is found (i.e. do we have the descriptors we need) 
    mask_X = np.array([True if i in y_labels_comp else False for i in y_labels_exp])

    # compares the two arrays, if same value is True in both then value = True in mask, otherwise = False.
    # i.e. does the ligand have an experimental result and descriptors
    mask = mask_y&mask_X

    # ligands_removed is a list of ligands that had zero-values
    count = 0
    ligands_removed = []
    for i in list(mask):
        if not i:
            ligands_removed.append(y_labels_exp[count])
        count += 1

    st.markdown("## Preprocessing - nan/zero entries removed:")
    st.write("Number of entries in experimental file before removing empty cells: {}".format(len(y)))
    st.write("Removing {} entries with empty cells".format(len(y)-sum(mask)))
    st.write("Ligands removed:",ligands_removed) #update this to view SMILES or at least names of ligands

    # remove all nan values from y, leaving experimental results
    y = y[np.array(mask)]

    # Convert reaction data to binary 1/0 based on y_cut
    y = np.array([1 if result > y_cut else 0 for result in y])

    # cut y_labels to only have ids for ligands with results
    y_labels = y_labels_exp[np.array(mask)]

    # X = array of descriptor values for the ligands with experimental results
    X = np.asarray(compinp.loc[y_labels],dtype=np.float)


    ####################################################################################################################
    #
    # Univariate logistic regression
    #
    #
    ####################################################################################################################

    from sigman_logreg.logreg_stats import calc_mcfad

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