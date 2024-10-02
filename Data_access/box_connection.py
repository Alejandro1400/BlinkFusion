import json
from boxsdk import OAuth2, Client
import tkinter as tk
from tkinter import filedialog

# Function to load credentials
def load_credentials(file_path):
    # Obtain path file from file name

    credentials = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(' = ')
                credentials[key] = value.strip("'")
        return credentials
    except Exception as e:
        print(f"Failed to load credentials: {e}")
        return None


def box_signin(auth_file_path=None):
    # Try to find the auth_info file in the current directory
    credentials = load_credentials(auth_file_path)
    if credentials is None or not all(key in credentials for key in ['DEVELOPER_TOKEN']):
        print(f"Invalid credentials or '{auth_file_path}' file not found.")
        return None
    try:
        oauth2 = OAuth2(None, None, access_token=credentials['DEVELOPER_TOKEN'])
        box_client = Client(oauth2)
        print('Connected to Box as:', box_client.user().get().login)
        return box_client
    except Exception as error:
        print("Box authentication failed:", error)
        return None


def database_stats(box_client, root_dir='Filament Data'):
    try:
        data_folder = find_folder_in_box(box_client, root_dir)
        if not data_folder:
            print(f"No folder found with the name {root_dir}")
            return
    except Exception as e:
        print(f"Failed to find root folder '{root_dir}': {e}")
        return

    database = {
        'name': data_folder.name,
        'id': data_folder.id,
        'dates': []
    }

    # Navigate through folders: Date -> Sample -> Cell
    for date_folder in data_folder.get_items():
        if date_folder.type == 'folder':
            date_dict = {
                'date_name': date_folder.name,
                'date_id': date_folder.id,
                'samples': []
            }
            for sample_folder in date_folder.get_items():
                if sample_folder.type == 'folder':
                    sample_dict = {
                        'sample_name': sample_folder.name,
                        'sample_id': sample_folder.id,
                        'cells': []
                    }
                    for cell_folder in sample_folder.get_items():
                        if cell_folder.type == 'folder':
                            cell_dict = {
                                'cell_name': cell_folder.name,
                                'cell_id': cell_folder.id,
                                'files': []
                            }
                            for cell_file in cell_folder.get_items():
                                if cell_file.name.endswith('.tiff') or cell_file.name.endswith('Results.csv') or cell_file.name.endswith('Processed.csv'):
                                    cell_dict['files'].append({
                                        'file_name': cell_file.name,
                                        'file_id': cell_file.id,
                                        'file_type': 'tiff' if cell_file.name.endswith('.tiff') else 'csv'
                                    })
                            sample_dict['cells'].append(cell_dict)
                    date_dict['samples'].append(sample_dict)
            database['dates'].append(date_dict)

    database_json = json.dumps(database, indent=4)
    
    return database_json


# Function to find a folder in Box recursively looking through the folders
def find_folder_in_box(box_client, folder_name, source_folder=None):
    try:
        if source_folder is None:
            root_folder = box_client.folder('0')
        else:
            root_folder = source_folder

        for item in root_folder.get_items():
            if item.name == folder_name:
                return item
            if item.type == 'folder':
                folder = find_folder_in_box(box_client, folder_name, source_folder=item)
                if folder:
                    return folder
        return None
    except Exception as e:
        print(f"Failed to find folder '{folder_name}': {e}")
        return None
    

def change_item_name(box_client, item_id, new_name, item_type='file'):
    """
    Change the name of a file or folder in Box.

    Args:
    box_client (Client): The Box client authenticated with the user's access token.
    item_id (str): The ID of the file or folder to rename.
    new_name (str): The new name for the file or folder.
    item_type (str): Type of the item, either 'file' or 'folder'.

    Returns:
    None
    """
    try:
        if item_type == 'file':
            item = box_client.file(file_id=item_id)
        elif item_type == 'folder':
            item = box_client.folder(folder_id=item_id)
        else:
            print("Invalid item type specified. Use 'file' or 'folder'.")
            return

        updated_item = item.update_info(data={'name': new_name})
        print(f"{item_type.capitalize()} name changed to: {updated_item.name}")
    except Exception as e:
        print(f"Failed to rename {item_type}: {e}")    