import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

from Dashboard.data_management import load_and_prepare_data
from Dashboard.graphs import create_boxplot, create_histogram, create_radar_chart
from Dashboard.metrics import calculate_summarized_metrics, calculate_summarized_metrics_2
from Data_access.file_explorer import *

# Constants
ALL_COLUMNS = ['Sample', 'Cell', 'Network', 'Contour', 'Length', 'Line width', 'Intensity', 'Contrast', 'Sinuosity', 'Gaps']
COLUMN_FILTER = ['Network', 'Contour', 'Length', 'Line width', 'Intensity', 'Contrast', 'Sinuosity', 'Gaps']
COLUMN_GRAPH = COLUMN_FILTER + ['Netw/Cont', 'Gaps/Cont']
X_AXIS_BOX = ['Sample', 'Cell']

@st.cache_data
def prepare_data():
    data = load_and_prepare_data()
    data['Sample_Cell'] = data[['Sample', 'Cell']].agg(' '.join, axis=1)
    print('hola')
    return data


def main():
    st.set_page_config(layout="wide")
    data = prepare_data()  # Load data before starting Streamlit

    st.sidebar.header("Data Filtering")

    sample_cell = data['Sample_Cell'].unique().tolist()
    selected_sample_cell = st.sidebar.multiselect("Exclude Sample and Cell", sample_cell)
    data = data[~data['Sample_Cell'].isin(selected_sample_cell)]

    if data.empty:
        st.write("No data available.")
        return

    selected_columns = st.sidebar.multiselect('Select Columns to Filter', ALL_COLUMNS)
    filters = {}

    for col in selected_columns:
        if data[col].dtype in ['float64', 'int64']:
            # For numerical columns, use a slider
            range_val = st.sidebar.slider(f"Range for {col}", float(data[col].min()), float(data[col].max()),
                                          (float(data[col].min()), float(data[col].max())))
            filters[col] = range_val
        elif data[col].dtype == 'object':
            # For categorical columns, use a multiselect
            selected_vals = st.sidebar.multiselect(f"Select {col}", data[col].unique().tolist(), data[col].unique().tolist())
            filters[col] = selected_vals

    # Apply filters to data
    for col, value in filters.items():
        if isinstance(value, tuple):
            data = data[(data[col] >= value[0]) & (data[col] <= value[1])]
        elif isinstance(value, list):
            data = data[data[col].isin(value)]

    st.write("# Data Exploration Dashboard")


    # summarized metrics section
    st.write("## Summarized Metrics")
    col1, col2 = st.columns(2)
    type_to_analyze = col1.selectbox("Select Type to Analyze", ['Sample', 'Cell'], index=0)
    #columns_to_exclude = col2.multiselect("Select Columns to Exclude", COLUMN_FILTER)

    if not data.empty:
        summarized_data = calculate_summarized_metrics(data, type_to_analyze, "")
        summarized_data_2  = calculate_summarized_metrics_2(data, 'Sample', "")
        st.write(summarized_data_2)


    # Boxplot section
    st.write("## Boxplot")

    col3, col4, col5 = st.columns(3)
    
    x_label_box = col3.selectbox("Select X label for Boxplot", X_AXIS_BOX, index=0)
    y_label_box = col4.selectbox("Select Y label for Boxplot", COLUMN_GRAPH, index=0)
    outliers = col5.checkbox("Remove Outliers", value=False)
    fig_box = create_boxplot(data, x_label_box, y_label_box, outliers)
    if fig_box:
        st.plotly_chart(fig_box, use_container_width=True)


    # Histogram section
    st.write("## Histogram")

    # Do 3 columns for selection
    col6, col7, col8, col9 = st.columns(4)

    x_label_hist = col6.selectbox("Select X label for Histogram", COLUMN_GRAPH, index=0)
    sample = col7.selectbox("Select Sample for Histogram", data['Sample'].unique(), index=0)
    bin_count = col8.slider("Select number of bins for Histogram", 1, 50, 10)
    histtype = col9.selectbox("Select histtype for Histogram", ['bar', 'barstacked', 'step', 'stepfilled'], index=0)

    
    fig_hist = create_histogram(data, x_label_hist, bin_count, sample, histtype)
    if fig_hist:
        st.pyplot(fig_hist)


    # Display the filtered data
    #st.write("Raw Data", data)

    # Radar chart section
    st.write("## Radar Chart")

    fig_radar = create_radar_chart(summarized_data)
    if fig_radar:
        # Display the figure
        st.pyplot(fig_radar)


if __name__ == "__main__":
    main()

