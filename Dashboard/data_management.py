import pandas as pd
from Data_access.file_explorer import dashboard_data, find_data_folder, folders_for_dashboard


def load_and_prepare_data():
    """Load all processed data from the 'Data' folder for the Streamlit app."""
    data_folder = find_data_folder()
    all_data = pd.DataFrame()
    if data_folder:
        valid_folders = folders_for_dashboard(data_folder)  # Find valid folders with 'Processed.csv'
        if not valid_folders:
            print("No valid data folders found.")
        for folder in valid_folders:
                processed_data = dashboard_data(folder)  # Load data from each valid folder
                if not processed_data.empty:
                    all_data = pd.concat([all_data, processed_data], ignore_index=True)

    return all_data

def fix_data_types(data):
    # Delete the rows where class is '0'
    data = data[data['Class'] != '0']

    # Change the data types
    data['Sample'] = data['Sample'].astype(str)
    data['Network'] = data['Network'].astype(int)
    data['Contour'] = data['Contour'].astype(int)
    data['Length'] = data['Length'].astype(float)
    data['Line width'] = data['Line width'].astype(float)
    data['Intensity'] = data['Intensity'].astype(float)
    data['Contrast'] = data['Contrast'].astype(float)
    data['Sinuosity'] = data['Sinuosity'].astype(float)
    data['Gaps'] = data['Gaps'].astype(int)
    data['Class'] = data['Class'].astype(str)