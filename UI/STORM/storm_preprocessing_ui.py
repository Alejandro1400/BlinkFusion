import os
import pandas as pd
import streamlit as st

from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import aggregate_metadata_info, czi_2_tiff, extract_values_from_title, read_tiff_metadata


def load_storm_metadata(storm_folder, reload_trigger):
    """
    Load and process metadata from a specified STORM folder. This function identifies valid folders,
    extracts metadata from `.tif` files, and compiles the data into a structured format.

    Args:
        storm_folder (str): Path to the main folder containing STORM `.tif` files.
        reload_trigger (int): A trigger to reload data when a user makes changes.

    Returns:
        dict: A dictionary containing metadata items where each key is the metadata `id` and
              the value is a list of tuples with `value` and `type`.
    """
    # Notify user about the processing progress
    #st.info("Searching for valid folders...")

    # Find folders containing `.tif` files and exclude unnecessary folders
    valid_folders = find_valid_folders(
        storm_folder,
        required_files={'.tif'}
    )
    valid_folders = [folder for folder in valid_folders if not folder.endswith('ROIs')]

    if not valid_folders:
        st.warning("No valid folders found. Please ensure the folder structure is correct.")
        return {}

    database_metadata = {}
    total_folders = len(valid_folders)

    # Progress bar for visual feedback
    status_text = st.text(f"Processing {total_folders} folders...")
    progress_bar = st.progress(0)

    # Process each valid folder
    for index, folder in enumerate(valid_folders):
        #st.write(f"Processing folder: **{folder}**")

        # Find the first `.tif` file in the folder
        tif_file = find_items(
            base_directory=folder,
            item='.tif',
            is_folder=False,
            check_multiple=False,
            search_by_extension=True
        )

        if not tif_file:
            st.warning(f"No `.tif` file found in folder: {folder}. Skipping.")
            continue

        # Extract metadata from the `.tif` file
        metadata = read_tiff_metadata(tif_file, root_tag='pulsestorm')

        # Debugging feedback
        #st.write(f"Metadata extracted from `{tif_file}`: {metadata}")

        # Process each metadata item and populate the database
        for item in metadata:
            entry = (item['value'], item['type'])

            if item['id'] in database_metadata:
                # Avoid duplicate entries
                if entry not in database_metadata[item['id']]:
                    database_metadata[item['id']].append(entry)
            else:
                # Initialize a new key in the dictionary
                database_metadata[item['id']] = [entry]

        # Update the progress bar
        progress_bar.progress((index + 1) / total_folders, text=f"Processing folder {index + 1} of {total_folders}")

    # Clear progress indicator once processing is complete
    status_text.empty()
    progress_bar.empty()
    st.success("Metadata loading completed.")

    return database_metadata


def reload_metadata():
    """
    Increment the reload trigger in Streamlit session state to refresh data.
    """
    if 'reload_trigger' in st.session_state:
        st.session_state.reload_trigger += 1
    else:
        st.session_state.reload_trigger = 1
    st.success("Reload triggered.")


def run_storm_preprocessing_ui(storm_folder):
    """
    Streamlit UI for preprocessing STORM metadata. This interface allows users to upload files,
    view and edit metadata, and add new metadata entries for processing.

    Args:
        storm_folder (str): Path to the folder containing STORM `.tif` files and associated metadata.
    """

    # Initialize session state variables
    st.session_state.setdefault('selected_for_folder', ["Date"])
    st.session_state.setdefault('metadata_values', {})
    st.session_state.setdefault('show_add_form', False)
    st.session_state.setdefault('reload_trigger', 0)

    # Load metadata from the database
    db_metadata = load_storm_metadata(storm_folder, st.session_state.reload_trigger)

    # Input for uploading file or folder
    upload_path = st.text_input(
        "Enter the path to the folder or file you wish to upload.",
        help="Specify the path to the folder or file for which metadata needs to be added or updated."
    )

    # File Metadata Section
    with st.expander("File Metadata"):
        if upload_path:
            all_files_metadata = {}
            if os.path.isdir(upload_path):
                st.write("**Folder selected:** Metadata will apply to all files in the folder.")
                files = find_items(base_directory=upload_path, item='.czi', is_folder=False, check_multiple=True, search_by_extension=True)

                for file in files:
                    title_metadata = extract_values_from_title(os.path.basename(file))
                    all_files_metadata[file] = title_metadata

            elif os.path.isfile(upload_path):
                st.write("**File selected:** Metadata extracted from the title will be displayed.")
                title_metadata = extract_values_from_title(os.path.basename(upload_path))
                all_files_metadata[upload_path] = title_metadata

            # Summarize metadata
            summary_df = aggregate_metadata_info(all_files_metadata)
            st.write(f"**Files to upload:** {len(all_files_metadata)}")
            st.write("**Extracted Metadata:** (Count indicates how many files have values)")
            st.table(summary_df)
        else:
            st.warning("Please select a folder or file to continue.")

    # Database Metadata Section
    with st.expander("Database Metadata"):
        st.write(f"**Metadata found in Database: {storm_folder}**")
        if not db_metadata:
            st.info("No metadata found in the database.")
        else:
            for metadata_id, entries in db_metadata.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Create current values list for the selectbox
                        current_values = [entry[0] for entry in entries] + [None, "Add new..."]
                        current_selection = st.session_state.metadata_values.get(metadata_id, [(current_values[0], entries[0][1])])[0][0]

                        if current_selection not in current_values:
                            current_values.insert(0, current_selection)

                        selected_value = st.selectbox(
                            f"{metadata_id} ({entries[0][1]})",
                            current_values,
                            index=current_values.index(current_selection),
                            key=f"{metadata_id}_value_select"
                        )

                        if selected_value == "Add new...":
                            new_value = st.text_input(f"Enter new value for {metadata_id}", key=f"{metadata_id}_new_value")
                            if new_value:
                                st.session_state.metadata_values[metadata_id] = [(new_value, entries[0][1])]
                        elif selected_value is None:
                            st.session_state.metadata_values[metadata_id] = [(None, entries[0][1])]
                        else:
                            st.session_state.metadata_values[metadata_id] = [(selected_value, entries[0][1])]

                    with col2:
                        folder_selected = st.checkbox("Folder", key=f"{metadata_id}_folder")
                        if folder_selected and metadata_id not in st.session_state.selected_for_folder:
                            st.session_state.selected_for_folder.append(metadata_id)
                        elif not folder_selected and metadata_id in st.session_state.selected_for_folder:
                            st.session_state.selected_for_folder.remove(metadata_id)

    # Add Metadata Form Section
    with st.expander("Add Metadata"):
        with st.form(key='new_metadata_form'):
            new_id = st.text_input("Metadata ID", help="Enter a unique identifier for the metadata.")
            new_type = st.selectbox("Data Type", ['string', 'float', 'int'], help="Select the type of the metadata value.")
            new_value = st.text_input("Value", help="Enter the value for the metadata.")
            new_folder = st.checkbox("Mark as Folder", help="Check this if the metadata represents a folder.")

            if st.form_submit_button(label="Add Metadata"):
                if not new_id or not new_value:
                    st.error("Both ID and Value are required.")
                elif new_id in st.session_state.metadata_values:
                    st.error("This ID already exists. Please use a unique ID.")
                elif not validate_value_with_type(new_value, new_type):
                    st.error("The value does not match the selected data type.")
                else:
                    st.session_state.metadata_values[new_id] = [(new_value, new_type)]
                    if new_folder:
                        st.session_state.selected_for_folder.append(new_id)
                    st.success(f"Metadata `{new_id}` added successfully!")
        

    with st.expander("Preview"):
        # Metadata overview
        st.subheader("Metadata Overview")
        st.info("The table below shows metadata with folder hierarchy based on selected metadata keys.")

        # Format metadata for display
        formatted_metadata = [{
            'root_tag': 'pulsestorm',
            'id': key,
            'type': values[0][1],  # Access type of the first tuple
            'value': values[0][0],  # Access value of the first tuple
            'Folder': (st.session_state.selected_for_folder.index(key) + 1 if key in st.session_state.selected_for_folder else None)
        } for key, values in st.session_state.metadata_values.items() if values[0][0] is not None]

        if formatted_metadata:
            df = pd.DataFrame(formatted_metadata)
            df = df.drop(columns=['root_tag'])
            df.set_index('id', inplace=True)
            df['Folder'] = df['Folder'].astype('Int64')  # Convert Folder column to Int64 for consistency
            st.table(df)
        else:
            st.warning("No metadata loaded for preview.")

        # Upload Files Button
        if st.button("Upload Files"):
            if all_files_metadata:
                num_files = len(all_files_metadata)
                st.success(f"Preparing to upload {num_files} file(s) to `{storm_folder}` with the following folder hierarchy:")

                # Display selected metadata keys for folder creation
                selected_metadata = [key for key in st.session_state.selected_for_folder]
                st.write(f"Metadata hierarchy: {selected_metadata}")

                # Remove 'Folder' key from formatted metadata for processing
                for entry in formatted_metadata:
                    entry.pop('Folder', None)

                # Process and upload files
                for file, metadata in all_files_metadata.items():
                    combined_metadata = formatted_metadata + metadata
                    st.write(f"Uploading file: `{file}` to `{storm_folder}`...")
                    
                    try:
                        # Convert CZI to TIFF with associated metadata
                        czi_2_tiff(file, storm_folder, st.session_state.selected_for_folder, combined_metadata)
                        st.success(f"File `{file}` successfully uploaded.")
                    except Exception as e:
                        st.error(f"Failed to upload file `{file}`. Error: {e}")
                        continue

                # Trigger metadata reload
                reload_metadata()
            else:
                st.warning("No files have been selected for upload.")


def validate_value_with_type(value, value_type):
    """
    Helper function to validate if the provided value matches the expected type.

    Args:
        value: The value to be validated.
        value_type (str): The expected type ('int', 'float', 'string').

    Returns:
        bool: True if the value matches the type, False otherwise.
    """
    try:
        if value_type == 'int':
            int(value)
        elif value_type == 'float':
            float(value)
        elif value_type == 'string':
            str(value)
        return True
    except ValueError:
        return False