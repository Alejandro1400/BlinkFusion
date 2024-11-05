from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import os

from Analysis.STORM.analytics_storm import obtain_molecules_metrics, calculate_duty_cycle, calculate_survival_fraction
from Dashboard.graphs import create_histogram, plot_intensity_vs_frame, plot_time_series_interactive
from Data_access.file_explorer import assign_structure_folders, find_items, find_valid_folders


@st.cache_data
def load_storm_data(pulseSTORM_folder):
    # List to store dataframes from all CSVs
    localizations = []
    tracks = []
    molecules = []
    metrics = []

    try:
        # Find the folder structure file (folder_structure.txt)
        structure_path = find_items(base_directory=pulseSTORM_folder, item='folder_structure.txt', is_folder=False)

        if structure_path:
            # Find all valid folders (folders that contain the required files)
            valid_folders = find_valid_folders(
                pulseSTORM_folder,
                required_files={'trackmate_locs_blink_stats.csv', 'trackmate_track_blink_stats.csv', 'trackmate_mol_blink_stats.csv'}
            )

            print(valid_folders)
            
            # Get the folder structure as a DataFrame
            image_df = assign_structure_folders(pulseSTORM_folder, structure_path, valid_folders)

            # Iterate through each valid folder
            for folder in valid_folders:
                locs = find_items(
                    base_directory=folder, 
                    item='trackmate_locs_blink_stats.csv', 
                    is_folder=False, 
                    search_by_extension=True
                )

                track = find_items(
                    base_directory=folder,
                    item='trackmate_track_blink_stats.csv',
                    is_folder=False,
                    search_by_extension=True
                )

                mol = find_items(
                    base_directory=folder,
                    item='trackmate_mol_blink_stats.csv',
                    is_folder=False,
                    search_by_extension=True
                )

                if locs and track and mol:
                    try:
                        # Load the dataframes
                        locs_df = pd.read_csv(locs)
                        track_df = pd.read_csv(track)
                        mol_df = pd.read_csv(mol)
                        
                        # Calculate the relative path and assign it
                        relative_path = os.path.relpath(folder, pulseSTORM_folder)
                        locs_df['IDENTIFIER'] = relative_path
                        track_df['IDENTIFIER'] = relative_path
                        mol_df['IDENTIFIER'] = relative_path

                        # Obtain mol metrics
                        mol_metrics = obtain_molecules_metrics(mol_df)
                        mol_metrics['IDENTIFIER'] = relative_path  # Use IDENTIFIER for joining later

                        # Append the metrics to the list to be processed later
                        metrics.append(mol_metrics)

                        # Append the dataframes to the lists
                        localizations.append(locs_df)
                        tracks.append(track_df)
                        molecules.append(mol_df)
                    except Exception as e:
                        print(f"Failed to process files in {folder}. Error: {e}")

        # Combine all dataframes into a single dataframe for each type
        localizations_df, tracks_df, molecules_df = map(pd.concat, [localizations, tracks, molecules])

        # Combine all mol_metrics into a single dataframe
        metrics_df = pd.concat(metrics)

        # Merge mol_metrics with image_df using IDENTIFIER
        merged_metrics = pd.merge(metrics_df, image_df, on='IDENTIFIER', how='left')
        # Reorder columns to have image_df columns first, then metrics_df columns, excluding duplicates
        merged_metrics = merged_metrics[image_df.columns.tolist() + [col for col in merged_metrics.columns if col not in image_df.columns]]

        return localizations_df, tracks_df, molecules_df, image_df, merged_metrics

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None



# Function to run the PulseSTORM UI
def run_storm_dashboard_ui(pulseSTORM_folder):
    
    # Load the cached dataframe
    localizations, tracks, molecules, images, metrics = load_storm_data(pulseSTORM_folder)

    if localizations.empty or tracks.empty or molecules.empty:
        st.error("No data found in the folder. Please check the folder path.")
        return
    
    # Determine the shared columns between 'images' and 'metrics', excluding 'IDENTIFIER'
    desc_columns = set(images.columns).intersection(metrics.columns) - {'IDENTIFIER'}

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
    
    # Display Images loaded
    st.write("Images loaded:")
    # Reset the index and drop the old one to ensure it does not appear in the display
    metrics_no_id = metrics.drop(columns=['IDENTIFIER'])
    st.dataframe(metrics_no_id)

    # Selection box with state
    selected_id = st.selectbox("Select Image", images['IDENTIFIER'].unique(), index=0)

    # Filter dataframes based on the selection
    selected_localizations = localizations[localizations['IDENTIFIER'] == selected_id]
    selected_tracks = tracks[tracks['IDENTIFIER'] == selected_id]
    selected_molecules = molecules[molecules['IDENTIFIER'] == selected_id]

    # Time series values plot
    interval = 1000
    total_frames = 40000

    # Plot vs time 
    st.subheader("Time Plots")


    duty_cycles = calculate_duty_cycle(selected_molecules, selected_tracks, interval, total_frames) 
    survival_fraction = calculate_survival_fraction(selected_molecules, selected_tracks, interval, total_frames)

    # Generate the interactive plot
    interactive_fig = plot_time_series_interactive(duty_cycles, survival_fraction)

    # Display the plot in Streamlit
    st.plotly_chart(interactive_fig, use_container_width=True)


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



    


    
