import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from Dashboard.graphs import plot_time_series_interactive

def display_time_series_image(
    selected_timeseries, qe_start, qe_end, qe_dc, qe_sf
):

    col1, col2 = st.columns(2)

    time_series_columns = selected_timeseries.columns.drop('index')

    with col1:
        selected_y1_label = st.selectbox(
            'Y1 Axis:',
            options=list(time_series_columns),
            key='y1_image_select',
            index=0,
            help="Choose the primary metric for the Y1 axis."
        )

    with col2:
        available_legends = [col for col in time_series_columns if col != selected_y1_label]
        selected_y2_label = st.selectbox(
            'Y2 Axis:',
            options=list(available_legends),
            key='y2_image_select',
            index=0,
            help="Choose the secondary metric for the Y2 axis."
        )

    # Prepare time series data for plotting
    selected_columns = ['index', selected_y1_label, selected_y2_label]
    if not selected_timeseries.empty:
        time_metric = selected_timeseries[selected_columns].set_index('index').sort_index()

        # Align time indices to nearest 10 for cleaner visualization
        time_metric.index = time_metric.index.map(lambda x: int(np.ceil(x / 10) * 10))

        # Create and display the time series plot
        fig = plot_time_series_interactive(
            metric1_data=time_metric[selected_y1_label],
            metric2_data=time_metric[selected_y2_label],
            metric1_name=selected_y1_label,
            metric2_name=selected_y2_label,
            qe_start=qe_start,
            qe_end=qe_end,
            qe_dc=qe_dc,
            qe_sf=qe_sf
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display time series data as a DataFrame
        st.write("Time Series Data")
        st.dataframe(time_metric, use_container_width=True, height=200)