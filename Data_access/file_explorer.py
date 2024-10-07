import os
import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
from skimage import io
from Analysis.SOAC.analytics_ridge_pipeline import analyze_data


def load_data_processing(folder_path):
    """Check if the specified files exist in the folder_path without a processed file."""
    required_files = {'Results.csv', 'Junctions.csv', '.tif'}
    found_files = set()

    for file in os.listdir(folder_path):
        for req in required_files:
            if file.endswith(req):
                found_files.add(req)
            elif file.endswith('Processed.csv'):
                return False

    if found_files == required_files:
        return True
    
    return False

def load_data_soac(folder_path):
    """Check if the specified files exist in the folder_path without a processed file."""
    required_files = {'.tif'}
    found_files = set()

    for file in os.listdir(folder_path):
        for req in required_files:
            if file.endswith(req):
                found_files.add(req)
            elif file.endswith('snakes.csv') or file.endswith('junctions.csv'):
                return False

    if found_files == required_files:
        return True
    
    return False

def load_data_dashboard(folder_path):
    """Check if the specified files exist in the folder_path."""
    required_files = {'Processed.csv'}
    found_files = set()

    for file in os.listdir(folder_path):
        for req in required_files:
            if file.endswith(req):
                found_files.add(req)

    if found_files == required_files:
        return True
    
    return False


def folders_for_processing(folder_path):
    """Recursively search and return a list of valid sub-folders."""
    valid_folders = []
    if load_data_processing(folder_path):
        valid_folders.append(folder_path)

    for subdir in os.listdir(folder_path):
        sub_path = f"{folder_path}/{subdir}"
        if os.path.isdir(sub_path):
            valid_folders.extend(folders_for_processing(sub_path))  # Recursive call

    return valid_folders


def folders_for_soac(folder_path):
    """Recursively search and return a list of valid sub-folders."""
    valid_folders = []
    if load_data_soac(folder_path):
        valid_folders.append(folder_path)

    for subdir in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subdir)
        if os.path.isdir(sub_path):
            valid_folders.extend(folders_for_soac(sub_path))  # Recursive call

    return valid_folders


def folders_for_dashboard(folder_path):
    """Recursively search and return a list of valid sub-folders."""
    valid_folders = []
    if load_data_dashboard(folder_path):
        valid_folders.append(folder_path)

    for subdir in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subdir)
        if os.path.isdir(sub_path):
            valid_folders.extend(folders_for_dashboard(sub_path))  # Recursive call

    return valid_folders

def processing_data(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('Results.csv'):
            results = pd.read_csv(os.path.join(folder_path, file))
        elif file.endswith('Junctions.csv'):
            junctions = pd.read_csv(os.path.join(folder_path, file))
        elif file.endswith('.tif'):
            image = io.imread(os.path.join(folder_path, file))
    return results, junctions, image

def dashboard_data(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('Processed.csv'):
            processed = pd.read_csv(os.path.join(folder_path, file))

    # Eliminates a cloumn 'Sample' if it exists
    if 'Sample' in processed.columns:
        processed.drop('Sample', axis=1, inplace=True)

    # Obtain the subdirs after the Data folder
    subdirs = folder_path.split('\\')
    subdirs = subdirs[subdirs.index('Data')+1:]
    # The first value will be the Date
    processed.insert(0, 'Date', subdirs[0])
    # The second value will be the Sample
    processed.insert(1, 'Sample', subdirs[1])
    # The last value will be the cell
    processed.insert(2, 'Cell', subdirs[-1])
    
    return processed
          
def save_processed_data(results, original_folder_path, results_folder_path):
    # Obtain the subdirs comparing the original folder path and the results folder path
    subdirs = original_folder_path.split('/')
    results_subdirs = results_folder_path.split('/')
    subdirs = results_subdirs[len(subdirs):]
    # Obtain the last subdir as the filename
    filename = results_subdirs[-1]
    # Add the subdir as a prefix to the results joined by _
    results.insert(0, 'Sample', os.sep.join(subdirs))

    """Save the results to the specified folder path with 'filename' + Processed.csv."""
    results.to_csv(os.path.join(results_folder_path, f'{filename}_Processed.csv'), index=False)


def find_item(base_directory=None, item_name=None, is_folder=True, extension=None):
    """
    Search for a folder or file within the directory tree starting from the base directory.
    
    Args:
    base_directory (str): The starting directory for the search.
    item_name (str, optional): The name of the folder or file to find. If None, the first match will be returned.
    is_folder (bool): Flag indicating whether to search for a folder (True) or a file (False).
    extension (str, optional): The extension of the file to find. If specified, the search will target files with this extension.

    Returns:
    str: The full path to the folder or file if found.

    Raises:
    FileNotFoundError: If no folder or file matches the search criteria.
    ValueError: If more than one file matches the search criteria and item_name is not specified.
    """
    if base_directory is None:
        base_directory = os.getcwd()  # Use current working directory if no base is provided

    matches = []
    for root, dirs, files in os.walk(base_directory):
        # Check directories only if is_folder is True
        if is_folder:
            if item_name:
                if item_name in dirs:
                    matches.append(os.path.join(root, item_name))
            else:
                # If no item name specified, add first directory found to matches
                if dirs:
                    matches.append(os.path.join(root, dirs[0]))
            continue

        # Check files only if is_folder is False
        if not is_folder:
            # Filter files by extension if one is provided
            if extension:
                files = [f for f in files if f.endswith(f".{extension}")]
            if item_name:
                files = [f for f in files if f.split('.')[0] == item_name]

            matches.extend(os.path.join(root, f) for f in files)

    # Handle the cases of no matches or multiple matches
    if not matches:
        raise FileNotFoundError("No matching folder or file found within the project structure.")
    if len(matches) > 1 and not item_name:
        raise ValueError("Multiple files found. Specify an item name or use a different search criteria.")
    return matches[0]

