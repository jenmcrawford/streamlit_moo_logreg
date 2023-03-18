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

from sigman_logreg.sig_fun import plot_fit_1D,heatmap_logreg,data_prep_fun


## HOMEPAGE

st.markdown("# :cow2: Multi-Objective Optimization :cow2:")

# --------------------------------------------------------------------------------------------------------------------------------------------
# sidebar setup - init with session state
# --------------------------------------------------------------------------------------------------------------------------------------------

# using empty lists as placeholders for imports to avoid issues with initialization

expinp = []
compinp = []

#initialize import process
if "import_complete" not in st.session_state:
    st.session_state.import_complete = False

st.sidebar.markdown("## Import experimental data")
exp_up = st.sidebar.file_uploader("Results?",type=".xlsx",key="exp_uploader")

if st.session_state.exp_uploader:
    exp_sheet = 'Data' # data excel sheet name
    y_label_col_exp= 1 # 0-indexed, the number of the column with the ligand idss
    expinp = pd.read_excel(exp_up,exp_sheet,index_col=y_label_col_exp,engine='openpyxl') #feels like there's a better way -- not going to bother improving

par_opt = st.sidebar.radio("Do you need to upload your own parameter set?",("No","Yes"),key="param_upload")
st.sidebar.markdown("**Default parameter set is database of bisphosphine descriptors**")
#default parameters are bisphosphine, but this obviously is unnecessary
if par_opt == "No":
    compinp = pd.read_excel("Bisphosphine_Parameters.xlsx",sheet_name="symm_adapt_lowconf")
if par_opt == "Yes":
    par_df = st.sidebar.file_uploader("Computed parameter set",type=".xlsx") #maybe need to process to be in same form as standard compinp
    compinp = pd.read_excel(par_df)

# --------------------------------------------------------------------------------------------------------------------------------------------
# pre-process data from imports - using typical Sigman formatting for imported Excel sheets
# --------------------------------------------------------------------------------------------------------------------------------------------

if type(compinp) is not list: #use session state instead?

    st.markdown("### :white_check_mark: :white_check_mark: Double check your imports!")
    st.markdown("***Current assumption for parameter import is that there are two rows of header information***")

    col1, col2 = st.columns(2)

    with col1:
        y_label_col_comp = st.selectbox("Which column contains integer ligand IDs?",options=compinp.columns)
        compinp = compinp.set_index(y_label_col_comp)

        compinp.index = compinp.index.map(str)

        dropbool = st.radio("Any other columns that should be dropped? (i.e., a column that is mostly text)",("Yes","No"))

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


    st.markdown("#### What should be the cut-off value to determine class identity? What defines a hit/no-hit?")
    st.markdown("*Hint: look at the scale of your reaction output*")


    y_cut = st.number_input("Determine threshold value for hit/no-hit :scissors:",min_value=0.0,max_value=100.0)

    X_names, X_labels, X_all, y_labels_comp, X_labelname, X_labelname_dict = inp.param_preprocess(compinp)
    y_labels_exp, y, y_labels, y_og = inp.exp_preprocess(expinp,resp_label,y_labels_comp, y_cut)

# --------------------------------------------------------------------------------------------------------------------------------------------
# view if class imbalance
# --------------------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### :bar_chart: Original distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(y_og)

    ax1.set_xlabel("Reaction output")
    ax1.set_ylabel("Count")

    st.pyplot(fig1)

with col2:
    st.markdown("#### :robot_face: Class labels")
    fig2, ax2 = plt.subplots()
    ax2.hist(y)

    ax2.set_xticks([0,1])
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)

# #update session state with complete import
# if st.session_state.param_import == True and st.session_state.exp_import == True:
#     st.session_state.import_complete = True

# Direct copy and paste from Sigman repo, adding in streamlit widgets when needed for user input
# --------------------------------------------------------------------------------------------------------------------------------------------
# top univariate logistic regression
# --------------------------------------------------------------------------------------------------------------------------------------------
st.markdown("### :medal: Get top univariate logistic regression models")
# X = array of descriptor values for the ligands with experimental results
X = np.asarray(compinp.loc[y_labels],dtype=np.float)
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
st.write(results_1_param.head(10))

# --------------------------------------------------------------------------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------------------------------------------------------------------------

X_sel,y_sel,labels_sel,exclude = X,y,y_labels,[]
X_train, X_test, y_train, y_test, TS, VS = data_prep_fun(X,X_sel,y_sel)

split_df = pd.DataFrame(list(zip(TS,VS)),columns=["TS ID","VS ID"])
# st.markdown("#### :file_folder: Training/Test Set Labels")
# st.write(split_df)

# --------------------------------------------------------------------------------------------------------------------------------------------
# run logistic regression
# --------------------------------------------------------------------------------------------------------------------------------------------

from sigman_logreg.sig_fun import man_sel_feat,forward_step_logreg,logreg_prep

df_train, df_test, newcols, response = logreg_prep(X_train,y_train,X_test,y_test,skipfeatures=[])

st.markdown("#### Example of showing manual model")

features_x=('x386','x459')
man_sel_feat(X_train,X_test,y_train,y_test,X_labels,X_labelname,X_names,y_labels,TS,VS,df_train,df_test,features_x,annotate_test_set=False,annotate_training_set=False)