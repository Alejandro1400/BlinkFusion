import streamlit as st
import pandas as pd

from Data_access.storm_db import STORMDatabaseManager

def load_storm_metadata(_database):
        """
        Load distinct metadata from the database where `tag='pulsestorm'`, but only for metadata entries
        that have associated molecules. Also, count the number of experiments that contain molecule data.

        Returns:
            tuple:
                - dict: Metadata dictionary where keys are metadata names and values are lists of unique values.
                - int: Number of files (experiments) that have at least one molecule.
        """
        # Aggregation pipeline to fetch distinct metadata with molecules
        pipeline = [
            {"$match": {
                "time_series": {"$exists": True, "$ne": []}  # Ensure the experiment has associated time series data
            }},
            {"$unwind": "$metadata"},  # Unwind metadata array
            {"$group": {
                "_id": "$metadata.name",
                "uniqueValues": {"$addToSet": "$metadata.value"}
            }},
            {"$sort": {"_id": 1}},  # Sort by metadata name
            {"$facet": {
                "metadataInfo": [
                    # Pass all previous grouped data
                    {"$project": {"_id": 1, "uniqueValues": 1}}
                ],
                "count": [
                    # Count the distinct experiment documents that have time series
                    {"$count": "numExperimentsWithTimeSeries"}
                ]
            }}
        ]

        metadata_result = list(_database.experiments.aggregate(pipeline))
        

        database_metadata = {item['_id']: item['uniqueValues'] for item in metadata_result[0]['metadataInfo']}

        return database_metadata

def select_filter_columns(metadata_values):
    """Prompt user to select columns for filtering metadata."""
    st.info("Apply filters to metadata columns. All data is included by default.")

    return st.multiselect(
        'Select columns to filter by:',
        options=list(metadata_values.keys()),
        key='filter_select',
        help="Choose metadata columns to filter the dataset."
    )

def apply_selected_filters(selected_filter_columns, metadata_values):
    """Apply user-selected filters and return the filters."""
    selected_filters = {}
    for col in selected_filter_columns:
        unique_values = metadata_values[col]  # Assume getting unique values from some database or function call
        selected_values = st.multiselect(
            f"Filter {col}:",
            options=unique_values,
            default=unique_values,
            key=f'filter_{col}',
            help=f"Select specific values for filtering the column '{col}'."
        )
        selected_filters[col] = selected_values
    return selected_filters

@st.cache_data
def fetch_and_display_filtered_data(_database, selected_filters):
    """Fetch and display metadata based on selected filters."""
    if selected_filters:
        metadata_analysis = _database.get_metadata(selected_filters)  # Assuming a function 'get_metadata'
    else:
        metadata_analysis = _database.get_metadata()  # Assuming a function 'get_metadata'

    if metadata_analysis:
        return metadata_analysis
    else:
        st.warning("No data found for the selected filters.")

def display_filtered_metadata(metadata_analysis, selected_filter_columns):
    """Display the filtered metadata."""
    experiment_ids = list(metadata_analysis.keys())
    st.success(f"Number of experiments retrieved: {len(experiment_ids)}")
    st.markdown("___")
    st.write("### Data after Filtering:")
    st.write("Displaying unique files with their metadata values.")
    
    # Ensure selected columns include 'Experiment' (folder path)
    display_columns = ["Experiment"] + selected_filter_columns if selected_filter_columns else list(metadata_analysis.values())[0].keys()
    
    metadata_df = pd.DataFrame.from_dict(metadata_analysis, orient="index")
    metadata_df.index.name = "Experiment ID"
    st.dataframe(metadata_df[display_columns])
    return metadata_df, display_columns

@st.cache_resource
def get_pre_metrics(experiment_ids):
    # Fetch datasets related to experiment IDs
    storm_db = STORMDatabaseManager()
    grouped_molecules = storm_db.get_grouped_molecules_and_tracks(experiment_ids)
    time_series_dict = storm_db.get_grouped_time_series(experiment_ids)

    return grouped_molecules, time_series_dict
