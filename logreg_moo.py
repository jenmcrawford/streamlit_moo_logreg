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


## HOMEPAGE

st.markdown("# :cow2: Multi-Objective Optimization :cow2:")

# --------------------------------------------------------------------------------------------------------------------------------------------
# sidebar setup - init with session state
# --------------------------------------------------------------------------------------------------------------------------------------------

# using empty lists as placeholders for imports to avoid issues with initialization
expinp = []
compinp = []

st.sidebar.markdown("## Import experimental data")
exp_up = st.sidebar.file_uploader("Results?",type=".xlsx")

if exp_up:
    exp_sheet = 'Data' # data excel sheet name
    y_label_col_exp= 1 # 0-indexed, the number of the column with the ligand idss
    expinp = pd.read_excel(exp_up,exp_sheet,index_col=y_label_col_exp,engine='openpyxl') #feels like there's a better way -- not going to bother improving

par_opt = st.sidebar.radio("Do you need to upload your own parameter set?",("No","Yes"))
if par_opt == "No":
    compinp = pd.read_excel("Bisphosphine_Parameters.xlsx",sheet_name="symm_adapt_lowconf")
if par_opt == "Yes":
    par_df = st.sidebar.file_uploader("Computed parameter set",type=".xlsx") #maybe need to process to be in same form as standard compinp
    compinp = pd.read_excel(par_df)


y_cut = st.sidebar.number_input("Determine threshold value for hit/no-hit:",min_value=0.0,max_value=100.0)

# --------------------------------------------------------------------------------------------------------------------------------------------
# pre-process data from imports - using typical Sigman formatting for imported Excel sheets
# --------------------------------------------------------------------------------------------------------------------------------------------

if type(compinp) is not list: #use session state instead?

    st.markdown("### :white_check_mark: :white_check_mark: Double check your imports!")
    st.markdown("***Current assumption for parameter import is that there are two rows of header information***")

    col1, col2 = st.columns(2)

    with col1:
        y_label_col_comp = st.selectbox("Which column contains ligand IDs?",options=compinp.columns)
        compinp = compinp.set_index(y_label_col_comp)
        compinp.index = compinp.index.map(str)

        dropbool = st.radio("Any other columns that should be dropped?",("No","Yes"))

        if dropbool == "Yes":
            dropset = st.multiselect("Which ones?",compinp.columns)

            if len(dropset) > 0:
                compinp.drop(columns=dropset,inplace=True)

    with col2:
        if type(expinp) is not list:
            resp_label = st.selectbox(
                "What is the reaction output?",
                options=expinp.columns
            )
            # response_col = expinp.columns.get_loc(option)

    st.markdown("### Currently loaded descriptors and reaction information")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### :floppy_disk: Parameter Dataframe")
        st.write(compinp)
    with col4:
        st.markdown("#### :test_tube: Reaction Information")
        st.write(expinp)

    X_names, X_labels, X_all, y_labels_comp, X_labelname, X_labelname_dict = inp.param_preprocess(compinp)
    y_labels_exp, y, y_labels, y_og = inp.exp_preprocess(expinp,resp_label,y_labels_comp, y_cut)

    # X = array of descriptor values for the ligands with experimental results
    X = np.asarray(compinp.loc[y_labels],dtype=np.float)

# --------------------------------------------------------------------------------------------------------------------------------------------
# view if class imbalance
# --------------------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### :bar_chart: Original distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(y_og)

    st.pyplot(fig1)

with col2:
    st.markdown("#### :robot_face: Class labels")
    fig2, ax2 = plt.subplots()
    ax2.hist(y)

    st.pyplot(fig2)
