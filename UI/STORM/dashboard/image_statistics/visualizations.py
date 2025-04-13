import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from Analysis.STORM.Calculator.frequency_calculator import calculate_frequency
from Dashboard.graphs import plot_histograms, plot_time_series_interactive

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


def display_histograms_image(
    selected_molecules, selected_metridata, qe_start, qe_end, frames, exp
):
    # Create four columns for user controls
    col1, col2, col3, col4 = st.columns(4)

    # Option to select population type (Whole Population or QE Population)
    with col1:
        selected_option1 = st.radio(
            "Select the tracks frequency to display",
            ["Whole Population", "Quasi-Equilibrium Population"],
            help="Choose whether to analyze the whole population or only the quasi-equilibrium population."
        )

    # Option to group data by molecule or track
    with col2:
        selected_option2 = st.radio(
            "Select the grouping to display",
            ["By Molecule", "By Track"],
            help="Choose to group the data by molecules or tracks for histogram generation."
        )

    # Slider for adjusting the number of bins in histograms
    with col3:
        num_bins = st.slider(
            "Number of Bins",
            min_value=5,
            max_value=50,
            value=20,
            help="Adjust the number of bins for the histograms."
        )

    # Checkbox for removing outliers
    with col4:
        remove_outliers = st.checkbox(
            "Remove Outliers",
            value=False,
            help="Enable to exclude outliers from the histograms."
        )

    # Convert options to parameters for histogram calculation
    population_type = 'quasi' if selected_option1 == "Quasi-Equilibrium Population" else 'whole'
    grouping_type = 'molecule' if selected_option2 == "By Molecule" else 'track'

    # Calculate frequencies for histograms
    duty_cycle, photons, switching_cycles, on_time, classification = calculate_frequency(
        selected_qe_molecules=selected_molecules,
        qe_start=qe_start,
        qe_end=qe_end,
        frames=frames,
        exp=exp,
        population=population_type,
        metric=grouping_type
    )

    # Generate and display histograms
    plot_histograms(
        duty_cycle=duty_cycle,
        photons=photons,
        switching_cycles=switching_cycles,
        on_time=on_time,
        metrics=selected_metridata,
        remove_outliers=remove_outliers,
        num_bins=num_bins,
        metric_type=grouping_type
    )