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
st.sidebar.markdown("## Import experimental data")
exp_up = st.sidebar.file_uploader("Results?",type=".xlsx")
expinp = pd.read_excel(exp_up,header=2) #feels like there's a better way -- not going to bother improving
par_opt = st.sidebar.radio("Do you need to upload your own parameter set?",("Yes","No"))

if par_opt == "Yes":
    par_df = st.sidebar.file_uploader("Computed parameter set",type=".xlsx") #maybe need to process to be in same form as standard compinp
if par_opt == "No":
    compinp = pd.read_excel("Bisphosphine_Parameters.xlsx",sheet_name="symm_adapt_lowconf")

y_cut = st.sidebar.number_input("Determine threshold value for hit/no-hit:",min_value=0.0,max_value=100.0)

## Run through import
st.markdown("# :cow2: Multi-Objective Optimization :cow2:")

## Process imports

compinp = compinp.drop(['smiles'],axis=1)
par_start_col = 1
compinp.index = compinp.index.map(str)

col1, col2 = st.columns(2)

with col1:
    y_label_col_comp = st.selectbox("Which column contains ligand IDs?",options=compinp.columns)

with col2:
    option = st.selectbox(
        "What is the reaction output?",
        options=expinp.columns
    )



