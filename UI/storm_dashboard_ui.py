from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import os

from Analysis.STORM.analytics_storm import calculate_molecule_metrics, calculate_quasi_equilibrium, calculate_time_series_metrics, obtain_molecules_metrics
from Dashboard.graphs import create_histogram, plot_histograms, plot_intensity_vs_frame, plot_time_series_interactive
from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import read_tiff_metadata


@st.cache_data
def load_storm_data(pulseSTORM_folder):
    # List to store dataframes from all CSVs
    metadata = []
    localizations = []
    tracks = []
    qe_tracks = []
    molecules = []
    metrics = []
    timeseries = []

    try:
        
        # Find all valid folders (folders that contain the required files)
        valid_folders = find_valid_folders(
            pulseSTORM_folder,
            required_files={'.tif','locs_blink_stats.csv', 'track_blink_stats.csv', 'mol_blink_stats.csv'}
        )

        # Iterate through each valid folder
        for folder in valid_folders:
            tif_file = find_items(
                base_directory=folder,
                item='.tif',
                is_folder=False,
                search_by_extension=True
            )

            locs = find_items(
                base_directory=folder, 
                item='locs_blink_stats.csv', 
                is_folder=False, 
                search_by_extension=True
            )

            track = find_items(
                base_directory=folder,
                item='track_blink_stats.csv',
                is_folder=False,
                search_by_extension=True
            )

            mol = find_items(
                base_directory=folder,
                item='mol_blink_stats.csv',
                is_folder=False,
                search_by_extension=True
            )

            if locs and track and mol:
                try:
                    # Calculate the relative path and assign it
                    relative_path = os.path.relpath(folder, pulseSTORM_folder)

                    print(f"Processing files in {folder}")

                    pulsestorm_metadata = read_tiff_metadata(tif_file, root_tag = ['pulsestorm', 'czi-pulsestorm'])
                    metadata_dict = {item['id']: item['value'] for item in pulsestorm_metadata}
                    meta_df = pd.DataFrame([metadata_dict])
                    meta_df.columns = meta_df.columns.str.upper()
                    meta_df['IDENTIFIER'] = relative_path

                    frames = meta_df['FRAMES'][0]
                    exp = meta_df['EXPOSURE'][0]


                    locs_df = pd.read_csv(locs)
                    track_df = pd.read_csv(track)
                    mol_df = pd.read_csv(mol)

                    locs_df['IDENTIFIER'] = relative_path
                    track_df['IDENTIFIER'] = relative_path
                    mol_df['IDENTIFIER'] = relative_path

                    print("Calculatin")



                    time_df = calculate_time_series_metrics(mol_df, track_df, 50, frames, exp)

                    mol_metrics, qe_tracks_m = obtain_molecules_metrics(mol_df, track_df, time_df, exp)
                    print(f"Calculating QE for {folder}")
                    print(mol_metrics)

                    
                    mol_metrics['IDENTIFIER'] = relative_path
                    time_df['IDENTIFIER'] = relative_path 
                    qe_tracks_m['IDENTIFIER'] = relative_path  

                    # Append the metrics to the list to be processed later
                    metrics.append(mol_metrics)
                    timeseries.append(time_df)
                    qe_tracks.append(qe_tracks_m)

                    # Append the dataframes to the lists
                    metadata.append(meta_df)
                    localizations.append(locs_df)
                    tracks.append(track_df)
                    molecules.append(mol_df)

                except Exception as e:
                    print(f"Failed to process files in {folder}. Error: {e}")

        # Combine all dataframes into a single dataframe for each type
        metadata_df, localizations_df, tracks_df, qe_tracks_df, molecules_df, timeseries_df = map(pd.concat, [metadata, localizations, tracks, qe_tracks, molecules, timeseries])

        # Combine all mol_metrics into a single dataframe
        metrics_df = pd.concat(metrics)

        # Merge mol_metrics with metadata_df using IDENTIFIER
        merged_metrics = pd.merge(metrics_df, metadata_df, on='IDENTIFIER', how='left')
        # Reorder columns to have metadata_df columns first, then metrics_df columns, excluding duplicates
        merged_metrics = merged_metrics[metadata_df.columns.tolist() + [col for col in merged_metrics.columns if col not in metadata_df.columns]]

        return localizations_df, tracks_df, qe_tracks_df, molecules_df, metadata_df, merged_metrics, timeseries_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None, None



# Function to run the PulseSTORM UI
def run_storm_dashboard_ui(pulseSTORM_folder):
    
    # Load the cached dataframe
    localizations, tracks, qe_tracks, molecules, metadata, metrics, results = load_storm_data(pulseSTORM_folder)

    if localizations.empty or tracks.empty or molecules.empty:
        st.error("No data found in the folder. Please check the folder path.")
        return
    
    # Determine the shared columns between 'metadata' and 'metrics', excluding 'IDENTIFIER'
    desc_columns = metadata.columns

    # Use Streamlit's multiselect to let the user select which columns to group by
    selected_group_columns = st.multiselect('Select columns to group by:', list(desc_columns))

    if selected_group_columns:
        # Group the metrics dataframe by the selected columns
        grouped_metrics = metrics.groupby(selected_group_columns)
        
        # Example of showing aggregated data, change 'size()' to other aggregations as needed
        st.write("Aggregated data based on selected group columns:")
        display_data = grouped_metrics.size().reset_index(name='Count')
        st.dataframe(display_data)
    else:
        st.write("No columns selected for grouping.")
    
    # Display metadata loaded
    st.write("metadata loaded:")
    # Reset the index and drop the old one to ensure it does not appear in the display
    metrics_no_id = metrics.drop(columns=['IDENTIFIER'])
    st.dataframe(metrics_no_id)

    # Selection box with state
    selected_id = st.selectbox("Select Image", metadata['IDENTIFIER'].unique(), index=0)

    # Filter dataframes based on the selection
    selected_localizations = localizations[localizations['IDENTIFIER'] == selected_id]
    selected_tracks = tracks[tracks['IDENTIFIER'] == selected_id]
    selected_molecules = molecules[molecules['IDENTIFIER'] == selected_id]
    selected_metadata = metadata[metadata['IDENTIFIER'] == selected_id]
    selected_qe_tracks = qe_tracks[qe_tracks['IDENTIFIER'] == selected_id]
    
    selected_results = results[results['IDENTIFIER'] == selected_id]
    selected_metrics = metrics[metrics['IDENTIFIER'] == selected_id]   

    # Plot vs time 
    st.subheader("Time Plots")

    duty_cycles = selected_results['Duty Cycle']
    survival_fraction = selected_results['Survival Fraction']

    # Round up the indices to the nearest 10
    duty_cycles.index = duty_cycles.index.map(lambda x: int(np.ceil(x / 10) * 10))
    survival_fraction.index = survival_fraction.index.map(lambda x: int(np.ceil(x / 10) * 10))

    # Convert to pd.Series
    duty_cycles = pd.Series(duty_cycles)
    survival_fraction = pd.Series(survival_fraction)

    #Include only certain columns from metrics
    # From the identifier grab only after the last slash
    metrics['IDENTIFIER'] = metrics['IDENTIFIER'].str.split('/').str[-1]
    # Round up to nearest 10 both the QE Start and QE End
    metrics['QE Start'] = metrics['QE Start'].apply(lambda x: round(x / 10) * 10)
    metrics['QE End'] = metrics['QE End'].apply(lambda x: round(x / 10) * 10)
    st.write(metrics[['IDENTIFIER', 'DATE', 'SAMPLE', 'PARTICLE', 'Molecules', 'QE Start', 'QE Duty Cycle', 'QE Survival Fraction', 'QE Active Population', 'QE Switching Cycles per mol', 'QE Photons per SC', 'QE Mean Uncertainty', 'QE On Time per SC', 'QE End']])
    

    # Grab the QE start and End from selected_metrics and round up to nearest 10
    qe_start = selected_metrics['QE Start'].iloc[0]
    qe_end = selected_metrics['QE End'].iloc[0]
    qe_dc = selected_metrics['QE Duty Cycle'].iloc[0]
    qe_sf = selected_metrics['QE Survival Fraction'].iloc[0]
    qe_start = round(qe_start / 10) * 10
    qe_end = round(qe_end / 10) * 10

    # Obtain exposure time from metadata
    exp = selected_metadata['EXPOSURE'].iloc[0]

    # Remove from duty_cycles and survival_fraction the last 2 values
    duty_cycles = duty_cycles.iloc[:-2]
    survival_fraction = survival_fraction.iloc[:-2]

    # Generate the interactive plot
    interactive_fig = plot_time_series_interactive(duty_cycles, survival_fraction, qe_start, qe_end, qe_dc, qe_sf)

    # Display the plot in Streamlit
    st.plotly_chart(interactive_fig, use_container_width=True)

    duty_cycle, photons, switching_cycles, track_intensity_within_range = calculate_molecule_metrics(selected_qe_tracks, qe_start, qe_end, exp)

    plot_histograms(duty_cycle, photons, switching_cycles, track_intensity_within_range)

    # Plot histograms of the data
    st.subheader("Histograms")
    
    # Track on_time histogram
    st.write("On Time Histogram")
    # Select number of bins
    histogram = create_histogram(selected_tracks, 'ON_TIME', 5, 'On Time per burst')
    st.pyplot(histogram)


    # Ask select a molecule 
    selected_molecule = st.selectbox("Select Molecule", selected_molecules['MOLECULE_ID'].unique(), index=4)

    # Filter the tracks for those with that Mol ID
    selected_tracks = selected_tracks[selected_tracks['MOLECULE_ID'] == selected_molecule]
    # Filter the localizations for those with the selected tracks id
    selected_localizations = selected_localizations[selected_localizations['TRACK_ID'].isin(selected_tracks['TRACK_ID'])]

    # Prepare the DataFrame for plotting
    plot_data = pd.DataFrame({
        'INTENSITY': selected_localizations['INTENSITY'].values,
        'FRAME': selected_localizations['FRAME'].values
    })
    plot_data = plot_data.set_index('FRAME').sort_index()

    # Interpolate only one-frame gaps
    # Identify gaps by checking where the difference between consecutive indices is 2
    diff = plot_data.index.to_series().diff()  # Compute the difference between consecutive frames
    gaps = diff[diff == 2].index  # Find indices where the difference is exactly 2

    # Calculate the mean for one-frame gaps
    for gap_start in gaps:
        if gap_start + 1 in plot_data.index:  # Ensure that gap_start + 1 is a valid index
            continue  # Skip if there's no actual gap
        # Calculate the mean of the intensities around the gap
        mean_intensity = (plot_data.at[gap_start - 1, 'INTENSITY'] + plot_data.at[gap_start + 1, 'INTENSITY']) / 2
        plot_data.at[gap_start, 'INTENSITY'] = mean_intensity  # Set the intensity for the gap

    # Fill other gaps with NaN to avoid artificial zeros, which could be misleading in data analysis
    plot_data = plot_data.reindex(range(plot_data.index.min(), plot_data.index.max() + 1), fill_value=0)

    # Plotting with Matplotlib
    fig = plot_intensity_vs_frame(plot_data)
    st.plotly_chart(fig, use_container_width=True)



    


    
