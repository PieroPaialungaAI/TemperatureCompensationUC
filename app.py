#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:21:04 2023

@author: paialupo
"""
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from utils import *

def main():
    st.title("Location Specific Temperature Compensation App for Joe")

    # Display file upload widget and process the uploaded file
    uploaded_file = st.file_uploader("Upload your Excel data file",type='xlsx')
    if uploaded_file is not None:
        data = import_data(uploaded_file)
        data_dict = preprocess_data(data)
        modelled_data = model_data(data_dict)
        # Allow the user to download the results
        st.download_button("Download Results", data=modelled_data.to_csv(), file_name='result.csv')
        
if __name__ == "__main__":
    main()
