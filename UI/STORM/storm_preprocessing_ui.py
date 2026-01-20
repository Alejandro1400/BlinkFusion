import os
import pandas as pd
import streamlit as st

from Data_access.storm_db import STORMDatabaseManager
from Data_access.file_explorer import find_items
from Data_access.metadata_manager import (
    aggregate_metadata_info, czi_2_tiff, extract_values_from_title
)


# ---------- DATABASE FUNCTIONS ---------- #
@st.cache_data
def load_storm_metadata(database_folder, reload_trigger):
    """
    Load metadata from the STORM database using the DatabaseManager class.
    """
    storm_db = STORMDatabaseManager()  # Initialize DB manager
    database_metadata = storm_db.load_storm_metadata()  # Fetch metadata from DB
    return database_metadata


def save_metadata_to_database(database_folder, all_files_metadata, formatted_metadata):
    """
    Saves metadata extracted from files into the database and converts CZI to TIFF while displaying real-time progress updates.

    Args:
        database_folder (str): The folder where metadata will be stored.
        all_files_metadata (dict): Dictionary containing file paths as keys and extracted metadata as values.
        formatted_metadata (list): List of metadata that was selected in the UI.
    """
    storm_db = STORMDatabaseManager()  # Initialize DB Manager
    progress_placeholder = st.empty()  # Placeholder for progress messages

    for index, (file, metadata) in enumerate(all_files_metadata.items(), start=1):
        file_name = os.path.basename(file)

        try:
            # Step 1: Show progress update
            progress_placeholder.write(f"**🔄 Processing file {index}/{len(all_files_metadata)}: `{file_name}`...**")

            # Combine formatted metadata with extracted metadata
            combined_metadata = formatted_metadata + metadata

            # Step 2: Converting CZI to TIFF
            progress_placeholder.write(f"🔄 **Converting `{file_name}` from CZI to TIFF...**")
            final_folder_path = czi_2_tiff(file, database_folder, st.session_state.selected_for_folder, combined_metadata)

            file_name = os.path.basename(final_folder_path)  # Update file name with new TIFF name

            # Extract relative folder path (removing database folder)
            relative_folder_path = os.path.relpath(final_folder_path, database_folder) + f"\{file_name}" + ".tif"

            # Step 3: Saving metadata to the database
            progress_placeholder.write(f"💾 **Saving metadata for `{file_name}` into the database...**")
            storm_db.save_metadata(combined_metadata, file_name, relative_folder_path)

            # Step 4: Success message
            progress_placeholder.success(f"✅ `{file_name}` successfully uploaded and metadata stored!")

        except Exception as e:
            progress_placeholder.error(f"❌ Failed to process `{file_name}`. Error: {e}")
            continue

    reload_metadata()  # Refresh UI to reflect changes




# ---------- UI FUNCTIONS ---------- #
def reload_metadata():
    """
    Increment the reload trigger in Streamlit session state to refresh data.
    """
    if 'reload_trigger' in st.session_state:
        st.session_state.reload_trigger += 1
    else:
        st.session_state.reload_trigger = 1
    st.success("Reload triggered.")


@st.cache_data
def display_file_metadata_section(upload_path):
    """
    Displays file metadata in the UI.
    """
    all_files_metadata = {}
    with st.expander("File Metadata"):
        if upload_path:
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

            summary_df = aggregate_metadata_info(all_files_metadata)
            st.write(f"**Files to upload:** {len(all_files_metadata)}")
            st.write("**Extracted Metadata:** (Count indicates how many files have values)")
            st.table(summary_df)
        else:
            st.warning("Please select a folder or file to continue.")

    return all_files_metadata


def display_database_metadata_section(database_metadata, storm_folder):
    """
    Displays database metadata in the UI.
    """
    with st.expander("Database Metadata"):
        st.write(f"**Metadata found in Database: {storm_folder}**")

        if not database_metadata:
            st.info("No metadata found in the database.")
        else:
            for metadata_id, entries in database_metadata.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
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


def display_add_metadata_section():
    """
    UI for adding new metadata manually.
    """
    with st.expander("Add Metadata"):
        with st.form(key="new_metadata_form"):
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


def display_preview_section(all_files_metadata, storm_folder):
    """
    Displays metadata preview and upload button.
    """
    with st.expander("Preview"):
        st.subheader("Metadata Overview")
        st.info("The table below shows metadata with folder hierarchy based on selected metadata keys.")

        # Format metadata for display
        formatted_metadata = [{
            'tag': 'pulsestorm',
            'id': key,
            'type': values[0][1],
            'value': values[0][0],
            'Folder': (st.session_state.selected_for_folder.index(key) + 1 if key in st.session_state.selected_for_folder else None)
        } for key, values in st.session_state.metadata_values.items() if values[0][0] is not None]

        if formatted_metadata:
            df = pd.DataFrame(formatted_metadata).drop(columns=['tag']).set_index('id')
            df['Folder'] = df['Folder'].astype('Int64')
            st.table(df)
        else:
            st.warning("No metadata loaded for preview.")

        # Upload Files Button
        if st.button("Upload Files"):
            if all_files_metadata:
                save_metadata_to_database(storm_folder, all_files_metadata, formatted_metadata)
            else:
                st.warning("No files have been selected for upload.")



def run_storm_preprocessing_ui(database_folder):
    """
    Streamlit UI for preprocessing STORM metadata.
    """
    st.session_state.setdefault("selected_for_folder", ["Date"])
    st.session_state.setdefault("metadata_values", {})
    st.session_state.setdefault("reload_trigger", 0)

    db_metadata = load_storm_metadata(database_folder, st.session_state.reload_trigger)
    upload_path = st.text_input("Enter the path to the folder or file you wish to upload.")

    all_files_metadata = display_file_metadata_section(upload_path)
    display_database_metadata_section(db_metadata, database_folder)
    display_add_metadata_section()
    display_preview_section(all_files_metadata, database_folder)


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