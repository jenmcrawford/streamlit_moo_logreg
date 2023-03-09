import numpy as np
import pandas as pd

import streamlit as st

def param_preprocess(compinp):
    #general labeling to ensure consistency between compinp and expinp
    X_names = list(compinp.iloc[0,:])
    compinp.drop(index=compinp.index[0],inplace=True) #dropping 2nd header row with actual descriptor names

    X_labels = list(compinp.columns)
    X_all = np.asarray(compinp[X_labels],dtype=np.float)
    y_labels_comp = np.asarray(list(compinp.index),dtype=str)

    X_labelname = [" ".join(i) for i in zip(X_labels,X_names)]
    X_labelname_dict = dict(zip(X_labels,X_names))

    #handle missing descriptors
    compnan = np.isnan(X_all).any(axis=1)
    y_labels_comp,X_all = y_labels_comp[~compnan],X_all[~compnan]

    return X_names, X_labels, X_all, y_labels_comp, X_labelname, X_labelname_dict

def exp_preprocess(expinp,resp_label,y_labels_comp,y_cut):
    # array of the experimental results
    y = np.asarray(expinp[resp_label],dtype=np.float)

    # array of all the ligand ids in the experimental file (curated below to give ligands with results only)
    y_labels_exp = np.asarray(list(expinp.index),dtype=str)

    # drop missing or nan ligands
    mask_y = ~np.isnan(y)
    mask_X = np.array([True if i in y_labels_comp else False for i in y_labels_exp])
    mask = mask_y&mask_X

    count = 0
    ligands_removed = []
    for i in list(mask):
        if not i:
            ligands_removed.append(y_labels_exp[count])
        count += 1

    y_og = y[np.array(mask)]
    y = np.array([1 if result > y_cut else 0 for result in y_og])

    y_labels = y_labels_exp[np.array(mask)]

    return y_labels_exp, y, y_labels, y_og

def class_cut(y,y_cut):
    y_class = np.array([1 if result > y_cut else 0 for result in y])
    return y_class