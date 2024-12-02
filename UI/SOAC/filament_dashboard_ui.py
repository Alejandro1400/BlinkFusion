# soac_filament_analysis.py
from math import nan
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from Analysis.SOAC.analytics_soac_filaments import calculate_snake_metrics, obtain_cell_metrics
from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import read_tiff_metadata


@st.cache_data
def load_filament_data(soac_folder):
    """
    Loads filament data from the specified SOAC folder. Processes metadata and snake metrics
    from valid folders containing `soac_results.csv` and `.tif` files.

    Args:
        soac_folder (str): Path to the SOAC folder containing analysis results.

    Returns:
        tuple: A tuple containing:
            - metadata_df (pd.DataFrame): Combined metadata dataframe.
            - snakes_df (pd.DataFrame): Combined snake metrics dataframe.
        Returns (None, None) if an error occurs.
    """

    metadata = []  # List to store metadata dataframes
    snakes = []    # List to store snake metrics dataframes

    # Initialize progress indicators
    status_text = st.text("Loading filaments data...")
    progress_bar = st.progress(0, text="Loading filaments data...")

    try:
        # Find all valid folders containing required files
        valid_folders = find_valid_folders(
            soac_folder,
            required_files={'soac_results.csv'}
        )
        total_folders = len(valid_folders)

        if total_folders == 0:
            st.warning("No valid folders found in the specified SOAC folder.")
            status_text.empty()
            progress_bar.empty()
            return None, None

        # Process each valid folder
        for index, folder in enumerate(valid_folders):
            time.sleep(0.01)  # Allow UI updates for the progress bar

            # Locate required files in the folder
            cell_file = find_items(
                base_directory=folder,
                item='soac_results.csv',
                is_folder=False,
                search_by_extension=True
            )
            tif_file = find_items(
                base_directory=folder,
                item='.tif',
                is_folder=False,
                search_by_extension=True
            )

            if cell_file and tif_file:
                try:
                    # Calculate relative path and load metadata
                    relative_path = os.path.relpath(folder, soac_folder)
                    pulsestorm_metadata = read_tiff_metadata(
                        tif_file, root_tag=['pulsestorm', 'tif-pulsestorm']
                    )

                    # Convert metadata into a dictionary
                    metadata_dict = {
                        item['id']: item['value'] for item in pulsestorm_metadata
                    }

                    # Create a metadata dataframe
                    meta_df = pd.DataFrame([metadata_dict])
                    meta_df['IDENTIFIER'] = relative_path
                    meta_df['IMAGE'] = os.path.basename(tif_file).split('.tif')[0].split('_')[0]

                    # Adjust pixel size if units are 'pixel'
                    if meta_df['Pixel Size Units'].values[0] == 'pixel':
                        meta_df['Pixel Size Units'] = 'um'
                        meta_df['Pixel Size X'] = 0.1079
                        meta_df['Pixel Size Y'] = 0.1079

                    # Load the SOAC results CSV
                    cell_df = pd.read_csv(cell_file)

                    # Calculate snake metrics
                    snk_df = cell_df.groupby(['File', 'Snake']).apply(calculate_snake_metrics).reset_index(drop=True)
                    snk_df['Length'] = snk_df['Length'] * meta_df['Pixel Size X'].values[0]
                    snk_df['IDENTIFIER'] = relative_path

                    # Append processed data to respective lists
                    metadata.append(meta_df)
                    snakes.append(snk_df)

                    # Update progress bar every 5 folders
                    if index % 5 == 0:
                        progress_bar.progress(
                            (index + 1) / total_folders,
                            text=f"Processing folder {index + 1}/{total_folders}"
                        )

                except Exception as e:
                    continue

        # Combine all dataframes into single dataframes
        metadata_df = pd.concat(metadata).reset_index(drop=True)
        snakes_df = pd.concat(snakes).reset_index(drop=True)

        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()

        return metadata_df, snakes_df

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        status_text.empty()
        progress_bar.empty()
        return None, None


def run_filament_dashboard_ui(soac_folder):
    """
    Displays the Filament Dashboard UI for visualizing and filtering filament data.

    Args:
        soac_folder (str): Path to the SOAC folder containing filament data.
    """

    if not soac_folder or not os.path.exists(soac_folder):
        st.error("SOAC folder not found. Please check the folder path and try again.")
        return
    
    # Load metadata and snakes datasets
    metadata, snakes = load_filament_data(soac_folder)

    # Check if data was loaded successfully
    if metadata is None or snakes is None:
        st.error("Failed to load data. Please check the folder structure and contents.")
        return

    # Display the number of images loaded
    st.write(f" **{len(metadata)}** images loaded in the dataset.")

    # Check if either dataset is empty
    if metadata.empty or snakes.empty:
        st.error("No data found. Ensure the folder contains valid files and try again.")
        return

    # Rename columns for clarity in the snakes dataset
    snakes = snakes.rename(columns={'Length': 'Length (um)', 'Intensity': 'Intensity (au)'})

    # Create copies of the datasets for analysis and filtering
    metadata_analysis = metadata.copy()
    snakes_analysis = snakes.copy()

    desc_columns = [col for col in metadata_analysis.columns if col != 'IDENTIFIER']

    # Apply Metadata Filters
    with st.expander("Filter Metadata Options", expanded=True):
        st.info("Apply filters to metadata based on the selected columns. All data is included by default.")

        # Select columns to apply filters
        selected_filter_columns = st.multiselect(
            'Select columns to filter by:',
            list(metadata_analysis.columns),
            key='filter_select',
            help="Choose metadata columns to filter the dataset."
        )

        if selected_filter_columns:
            filter_mask_md = pd.Series([True] * len(metadata_analysis), index=metadata_analysis.index)

            for col in selected_filter_columns:
                unique_values = metadata_analysis[col].unique()
                unique_values_list = [x if not pd.isnull(x) else 'nan' for x in unique_values]

                selected_values = st.multiselect(
                    f"Filter {col}:",
                    unique_values_list,
                    default=unique_values_list,
                    key=f'filter_{col}'
                )

                selected_values = [np.nan if v == 'nan' else v for v in selected_values]

                filter_mask_md &= metadata_analysis[col].isin(selected_values) | (
                    metadata_analysis[col].isnull() & (np.nan in selected_values)
                )

            metadata_analysis = metadata_analysis[filter_mask_md].reset_index(drop=True)

            if metadata_analysis.empty:
                st.warning("No data found for the selected metadata filters.")
                return

            # Filter snakes based on filtered metadata
            identifiers = metadata_analysis['IDENTIFIER'].unique()
            snakes_analysis = snakes[snakes['IDENTIFIER'].isin(identifiers)].reset_index(drop=True)

    snakes_metrics_analysis = snakes_analysis.copy()

    # Apply Range Filters
    with st.expander("Filter Data", expanded=True):
        st.info("Use this section to filter snake metrics based on numeric columns. All values are selected by default.")

        numeric_columns = snakes_analysis.select_dtypes(include=['number']).columns
        numeric_columns = [col for col in numeric_columns if col != 'IDENTIFIER']

        selected_filter_columns = st.multiselect(
            'Select metrics to filter by:',
            numeric_columns,
            key='range_select',
            help="Choose numeric metrics to apply range-based filters."
        )

        if selected_filter_columns:
            filter_mask_snakes = pd.Series([True] * len(snakes_analysis), index=snakes_analysis.index)

            for col in selected_filter_columns:
                col_min, col_max = snakes_analysis[col].min(), snakes_analysis[col].max()
                selected_range = st.slider(
                    f"Select range for {col}:",
                    min_value=float(col_min),
                    max_value=float(col_max),
                    value=(float(col_min), float(col_max)),
                    key=f'slider_{col}'
                )

                filter_mask_snakes &= snakes_analysis[col].between(selected_range[0], selected_range[1])

            snakes_metrics_analysis = snakes_analysis[filter_mask_snakes].reset_index(drop=True)

            if snakes_metrics_analysis.empty:
                st.warning("No snakes remaining after range filters.")
                return

            # Update metadata based on filtered snakes
            identifiers = snakes_metrics_analysis['IDENTIFIER'].unique()
            metadata_analysis = metadata_analysis[metadata_analysis['IDENTIFIER'].isin(identifiers)].reset_index(drop=True)
        else:
            st.warning("No filters applied. Showing all data.")

        # Display statistics about filtered snakes
        filtered_snakes_count = len(snakes_analysis) - len(snakes_metrics_analysis)
        st.write(f"Filtered out **{filtered_snakes_count}** snakes from **{len(snakes_analysis)}** total snakes.")

        # Snake count statistics per image
        if not snakes_metrics_analysis.empty:
            snake_count = snakes_metrics_analysis.groupby('IDENTIFIER').size()
            st.write(f"Average Number of snakes per image: {int(snake_count.mean())}")
            st.write(f"Minimum Number of snakes per image: {int(snake_count.min())}")
            st.write(f"Maximum Number of snakes per image: {int(snake_count.max())}")
        else:
            st.warning("No snakes remaining after filtering.")

    # Merge filtered metadata and snake metrics for further analysis
    if not snakes_metrics_analysis.empty and not metadata_analysis.empty:
        metridata = pd.merge(
            metadata_analysis, 
            snakes_metrics_analysis, 
            on='IDENTIFIER', 
            how='inner'
        )

    # Group by metrics to calculate means and standard deviations
    snakes_group_columns = []
    for column in snakes_metrics_analysis.columns:
        if column != 'IDENTIFIER':
            snakes_group_columns.append(f"{column} - mean")
            snakes_group_columns.append(f"{column} - std")
    

    with st.expander("Data Analysis", expanded=True):
        # Create two columns for grouping and metric selection
        col1, col2 = st.columns(2)

        with col1:
            selected_group_columns = st.multiselect(
                'Select columns to group by:',
                list(metadata_analysis.columns),
                key='group_select',
                help="Choose metadata columns to group the data. Default is 'IDENTIFIER'."
            )

        with col2:
            selected_metric_columns = st.multiselect(
                'Select columns to calculate metrics for:',
                snakes_group_columns,
                key='metric_select',
                help="Choose metrics from the snake dataset to calculate (e.g., mean, std)."
            )

        if not snakes_metrics_analysis.empty:
            # Combine selected group and metric columns
            selected_columns = selected_group_columns + selected_metric_columns

            # Default to all columns if no columns are selected
            if not selected_columns:
                selected_columns = metridata.columns.tolist()

            # Ensure at least one group column is selected
            if not selected_group_columns:
                selected_group_columns = ['IDENTIFIER']
                st.warning("No group column selected. Defaulting to 'IDENTIFIER'.")

            # Calculate metrics based on selected group columns
            try:
                metrics = obtain_cell_metrics(
                    metridata, selected_group_columns
                ).set_index(selected_group_columns)[selected_metric_columns]

                # Display grouped metrics
                st.write("### Data after Grouping")
                st.dataframe(metrics)
            except Exception as e:
                st.error(f"An error occurred while grouping or calculating metrics: {e}")
        else:
            st.warning("Snake metrics data is empty. Please apply appropriate filters or check your dataset.")

    
    with st.expander("Comparison Analysis", expanded=True):
        """
        Allows users to compare metrics from the dataset using various plot types.
        Users can select X-axis, Y-axis, Legend, and the type of plot to visualize the data.
        """

        # Create four columns for plot customization
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            x_axis = st.selectbox(
                'Select X-axis:',
                metadata_analysis.columns,
                key='x_axis_select',
                help="Choose a column from metadata to use as the X-axis for the plot."
            )

        with col2:
            y_axis = st.selectbox(
                'Select Y-axis:',
                snakes_metrics_analysis.columns,
                key='y_axis_select',
                help="Choose a column from snake metrics to use as the Y-axis for the plot."
            )

        with col3:
            available_legend = [col for col in desc_columns if col != 'IDENTIFIER' and col != x_axis]
            legend_column = st.selectbox(
                'Select Legend:',
                options=['None'] + available_legend,
                key='legend_select',
                help="Choose a column to use as the legend for grouping data in the plot. Select 'None' for no grouping."
            )

        with col4:
            plot_type = st.selectbox(
                'Select Plot Type:',
                options=['Bar', 'Line', 'Scatter', 'Box', 'Violin', 'Histogram'],
                key='plot_type_select',
                help="Select the type of plot to visualize the data."
            )

        # Check if the merged dataset is not empty
        if not metridata.empty:

            # Plot based on the selected plot type
            if plot_type == 'Bar':
                # Group data by x_axis and calculate the mean for the y_axis
                if legend_column == 'None':
                    grouped_data = metridata.groupby(x_axis, as_index=False)[y_axis].mean()
                    fig = px.bar(
                        grouped_data,
                        x=x_axis,
                        y=y_axis,
                        title=f'Mean {y_axis} vs {x_axis}',
                        template='plotly_white',
                    )
                else:
                    grouped_data = metridata.groupby([x_axis, legend_column], as_index=False)[y_axis].mean()
                    fig = px.bar(
                        grouped_data,
                        x=x_axis,
                        y=y_axis,
                        color=legend_column,
                        barmode='group',
                        title=f'Mean {y_axis} vs {x_axis} grouped by {legend_column}',
                        template='plotly_white',
                    )

            elif plot_type == 'Line':
                if legend_column == 'None':
                    grouped_data = metridata.groupby(x_axis, as_index=False)[y_axis].mean()
                    fig = px.line(
                        grouped_data,
                        x=x_axis,
                        y=y_axis,
                        title=f'Mean {y_axis} vs {x_axis}',
                        template='plotly_white',
                    )
                else:
                    grouped_data = metridata.groupby([x_axis, legend_column], as_index=False)[y_axis].mean()
                    fig = px.line(
                        grouped_data,
                        x=x_axis,
                        y=y_axis,
                        color=legend_column,
                        title=f'Mean {y_axis} vs {x_axis} grouped by {legend_column}',
                        template='plotly_white',
                    )

            elif plot_type == 'Scatter':
                if legend_column == 'None':
                    fig = px.scatter(
                        metridata,
                        x=x_axis,
                        y=y_axis,
                        title=f'Scatter Plot of {y_axis} vs {x_axis}',
                        template='plotly_white',
                    )
                else:
                    fig = px.scatter(
                        metridata,
                        x=x_axis,
                        y=y_axis,
                        color=legend_column,
                        title=f'Scatter Plot of {y_axis} vs {x_axis} grouped by {legend_column}',
                        template='plotly_white',
                    )

            elif plot_type == 'Box':
                if legend_column == 'None':
                    fig = px.box(
                        metridata,
                        x=x_axis,
                        y=y_axis,
                        title=f'Box Plot of {y_axis} vs {x_axis}',
                        template='plotly_white',
                        points=False,
                    )
                else:
                    fig = px.box(
                        metridata,
                        x=x_axis,
                        y=y_axis,
                        color=legend_column,
                        title=f'Box Plot of {y_axis} vs {x_axis} grouped by {legend_column}',
                        template='plotly_white',
                        points=False,
                    )

            elif plot_type == 'Violin':
                if legend_column == 'None':
                    fig = px.violin(
                        metridata,
                        x=x_axis,
                        y=y_axis,
                        title=f'Violin Plot of {y_axis} vs {x_axis}',
                        template='plotly_white',
                        points=False,
                    )
                else:
                    fig = px.violin(
                        metridata,
                        x=x_axis,
                        y=y_axis,
                        color=legend_column,
                        title=f'Violin Plot of {y_axis} vs {x_axis} grouped by {legend_column}',
                        template='plotly_white',
                        points=False,
                    )

            elif plot_type == 'Histogram':
                fig = px.histogram(
                    metridata,
                    x=x_axis,
                    y=y_axis,
                    color=legend_column if legend_column != 'None' else None,
                    title=f'Histogram of {y_axis} grouped by {x_axis}' if legend_column != 'None' else f'Histogram of {y_axis}',
                    template='plotly_white',
                    barmode='overlay',
                )

            # Enhance layout for better readability
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                font=dict(family="Arial, sans-serif", size=14, color="black"),
                title_font=dict(size=16),
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            )
            st.plotly_chart(fig)

            st.info(f"On the Top Right while hovering over the graph there are zoom and capture options.")
        else:
            st.warning("No data available for comparison. Please check your filters or dataset.")





        

            

