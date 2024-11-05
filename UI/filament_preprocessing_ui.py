import streamlit as st

from Data_access.file_explorer import append_metadata_tags, find_items, find_valid_folders, parse_metadata_input, read_tiff_metadata

#@st.cache_data
def load_filament_metadata(soac_folder):

    #try: 
    valid_folders = find_valid_folders(
        soac_folder,
        required_files={'.tif', 'ridge_metrics.csv', 'soac_results.csv'}
    )

    for folder in valid_folders:
            
        # Find tif files in folder
        tif_file = find_items(base_directory=folder, item='.tif', is_folder=False, check_multiple=False, search_by_extension=True)

        # Obtain metadata from the tif file
        metadata = read_tiff_metadata('c://Users//usuario//Box//For Alejandro//Filament Data//8_12_24//A1//cell1//cell1.tif', root_tag = 'storm-prop')

        # Add tags to the metadata
        new_tif_path = 'modified.tif'
        root_tag = 'prop'
        metadata_dict = [
            {'id': 'Description', 'value': 'Sample Description Updated', 'type': 'string'},
            {'id': 'Version', 'value': '1.0', 'type': 'float'}
        ]

        #print(f"Metadata: {metadata}")
        print(f"File: {tif_file}")

        #append_metadata_tags(tif_file, new_tif_path, root_tag, metadata_dict)

        return metadata, tif_file
        
    #except Exception as e:
    #    print(f"Failed to process files in {folder}. Error: {e}")


def run_filament_preprocessing_ui(soac_folder):

    new_path = st.text_input("Enter the path to the folder or file you wish to upload.")
    
    metadata, tif_file = load_filament_metadata(soac_folder)

    st.write(metadata)
    
    with st.expander("Configure File Upload"):
        st.write("Here you can configure the metadata that will be added when uploading the file to your Database.")
        
    
        st.write(metadata)

        # Enter in a text_input box the metadata that will be added to the files
        metadata_input = st.text_input("Enter the metadata you wish to add to the file as Name, Type and Value (To add more than one use &). Example: Description, string, Sample Description & Version, float, 1.0")

        if st.button("Add Metadata"):
            if tif_file and metadata_input:
                tags = parse_metadata_input(metadata_input)
                try:
                    #add_tags_to_tiff(tif_file, tags)
                    st.success("Metadata added successfully.")
                except Exception as e:
                    st.error(f"Failed to add metadata: {e}")
            else:
                st.error("Please provide both a file path and metadata.")
        
    