import numpy as np
import pandas as pd

import streamlit as st

def param_preprocess(compinp):
    X_names = list(compinp.iloc[0,-1])

    return X_names

def exp_preprocess(compinp):
    pass

def class_cut(expinp,y_cut):
    pass