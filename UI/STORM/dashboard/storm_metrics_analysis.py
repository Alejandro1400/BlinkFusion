import pandas as pd
import streamlit as st

from Analysis.STORM.Calculator.molecule_metrics import MoleculeMetrics

st.cache_resource
def metrics_metadata_merge(grouped_molecules, time_series_dict, metadata_analysis, metadata_df):
    try:
        molecule_metrics = MoleculeMetrics(grouped_molecules, time_series_dict, metadata_analysis)
        # Obtain molecular metrics based on filtered tracks and timeseries
        metrics = molecule_metrics.obtain_molecules_metrics()
        metrics_df = pd.DataFrame(metrics)

        # Add a column for the number of images and align it at the beginning
        metrics_df.insert(0, '# Images', len(metadata_analysis))
        metrics_columns = metrics_df.columns.drop('Experiment ID')

        # Merge metrics with metadata for context and further analysis
        if not metrics_df.empty and not metadata_df.empty:
            metridata = pd.merge(metadata_df, metrics_df, on='Experiment ID', how='inner')
            st.write("Metrics successfully calculated and merged with metadata.")
            return metridata, metrics_columns
        else:
            st.warning("No metrics were calculated. Please check your data or filtering criteria.")
    except Exception as e:
        st.error(f"An error occurred while calculating metrics: {e}")


def display_blinking_statistics(metridata, desc_columns, metrics_columns, grouped_molecules, time_series_dict, metadata_analysis):
    """
    Display an expander with options for users to group metrics and display aggregated results.
    Args:
        metridata (DataFrame): DataFrame containing metrics data merged with metadata.
        desc_columns (list): List of description columns available for grouping.
        metrics_columns (list): List of metrics columns available for display.
        grouped_molecules: Data about grouped molecules.
        time_series_dict: Dictionary containing time series data.
        metadata_analysis: Metadata analysis results.
    """
    with st.expander("Blinking Statistics", expanded=True):
        """
        Section for grouping metrics and displaying aggregated results. 
        Users can group data by selected columns and choose metrics to display.
        """
        molecule_metrics = MoleculeMetrics(grouped_molecules, time_series_dict, metadata_analysis)

        # Create two columns for grouping and metric selection
        col1, col2 = st.columns(2)

        with col1:
            selected_group_columns = st.multiselect(
                'Choose columns to group by (Be mindful of the hierarchical order):',
                list(desc_columns),
                key='group_by_select',
                help="Select one or more metadata columns to group data hierarchically."
            )

        with col2:
            selected_metrics_columns = st.multiselect(
                'Choose columns to display:',
                list(metrics_columns),
                key='metrics_select',
                help="Select one or more metrics columns to display aggregated results."
            )

        # Display grouped data if metrics are available
        if not metridata.empty:
            # Combine selected group and metric columns
            selected_columns = selected_group_columns + selected_metrics_columns

            if not selected_columns:
                selected_columns = metridata.columns.tolist()  # Default to all columns

            if not selected_group_columns:
                selected_group_columns = ['Experiment']  # Default grouping column

            # Aggregate metrics based on group selections
            grouped_metrics = metridata.groupby(selected_group_columns).apply(molecule_metrics.aggregate_metrics)
            grouped_metrics = grouped_metrics[selected_metrics_columns]

            # Display the grouped metrics
            st.markdown("___")
            st.dataframe(grouped_metrics, height=300)