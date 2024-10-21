import os
import shutil
import pandas as pd
from skimage import io
from Analysis.SOAC.analytics_ridge_filaments import analyze_data
from PIL import Image

def load_data_ridge_detector_processing(folder_path):
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

def load_data_soac_preprocessing(folder_path):
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

def load_data_ridge_detector_analytics(folder_path):
    """Check if the specified files exist in the folder_path."""
    required_files = {'rd_proc.csv'}
    found_files = set()

    for file in os.listdir(folder_path):
        for req in required_files:
            if file.endswith(req):
                found_files.add(req)

    if found_files == required_files:
        return True
    
    return False

def check_data(folder_path, required_files, exclude_files=None, exclude_folders=None):
    """
    Check the specified folder for required files, count them, and handle excluded files and folders.
    
    :param folder_path: str, the path to the directory to check
    :param required_files: set, a set of filenames or extensions that are required
    :param exclude_files: set, a set of filenames or extensions to exclude
    :param exclude_folders: set, a set of folder names to exclude from being valid
    :return: dict, details about found required files and status
    """
    found_files = {}
    excluded_found = False
    excluded_folder_found = False

    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            for req in required_files:
                if file.endswith(req):
                    found_files[req] = found_files.get(req, 0) + 1
            if exclude_files and any(file.endswith(exc) for exc in exclude_files):
                excluded_found = True
        if exclude_folders and os.path.isdir(os.path.join(folder_path, file)) and file in exclude_folders:
            excluded_folder_found = True

    missing_files = required_files - found_files.keys()
    output = {'found_files': found_files}

    if excluded_found or excluded_folder_found:
        output['status'] = f"For {folder_path}: Processing already part of the database"
    elif missing_files:
        output['status'] = f"For {folder_path}: Missing files: {missing_files}"
    else:
        output['status'] = f"For {folder_path}: All required files found and counted"

    return output

def find_valid_folders(folder_path, required_files, exclude_files=None, exclude_folders=None):
    """
    Recursively search and return a list of valid sub-folders based on the check_data criteria.
    
    :param folder_path: str, the base directory to start the search
    :param required_files: set, a set of filenames or extensions that are required
    :param exclude_files: set, a set of filenames or extensions to exclude
    :return: list, a list of valid folders
    """
    valid_folders = []

    result = check_data(folder_path, required_files, exclude_files, exclude_folders)

    # Check if the current folder matches the criteria
    if result['status'].startswith(f"For {folder_path}: All required files found and counted"):
        valid_folders.append(folder_path)

    # Recursively check sub-folders
    for subdir in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subdir)
        # Skip excluded folders
        if exclude_folders and subdir in exclude_folders:
            continue
        if os.path.isdir(sub_path):
            valid_folders.extend(find_valid_folders(sub_path, required_files, exclude_files, exclude_folders))

    return valid_folders


def ridge_detector_folders_processing(folder_path):
    """Recursively search and return a list of valid sub-folders."""
    valid_folders = []
    if load_data_ridge_detector_processing(folder_path):
        valid_folders.append(folder_path)

    for subdir in os.listdir(folder_path):
        sub_path = f"{folder_path}/{subdir}"
        if os.path.isdir(sub_path):
            valid_folders.extend(ridge_detector_folders_processing(sub_path))  # Recursive call

    return valid_folders


def soac_folders_processing(folder_path):
    """Recursively search and return a list of valid sub-folders."""
    valid_folders = []
    if load_data_soac_preprocessing(folder_path):
        valid_folders.append(folder_path)

    for subdir in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subdir)
        if os.path.isdir(sub_path):
            valid_folders.extend(soac_folders_processing(sub_path))  # Recursive call

    return valid_folders


def ridge_detector_folders_analytics(folder_path):
    """Recursively search and return a list of valid sub-folders."""
    valid_folders = []
    if load_data_ridge_detector_analytics(folder_path):
        valid_folders.append(folder_path)

    for subdir in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subdir)
        if os.path.isdir(sub_path):
            valid_folders.extend(ridge_detector_folders_analytics(sub_path))  # Recursive call

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


def find_items(base_directory=None, item="Data", is_folder=True, check_multiple=False, search_by_extension=False):
    """
    Search for a folder or file within the directory tree starting from the base directory.
    Optionally checks for multiple occurrences of a specified file type or name, but stops searching
    further subdirectories once a match is found in a single directory.

    Args:
    base_directory (str): The starting directory for the search.
    item (str): The name of the folder or file to find, or the file extension to look for if search_by_extension is True.
    is_folder (bool): Flag indicating whether to search for a folder (True) or a file (False).
    check_multiple (bool): If True, search for multiple files of the given type or name in the first matching directory only.
    search_by_extension (bool): If True, search items by their extension, otherwise by their name.

    Returns:
    str or list: The full path to the folder or file if found, or a list of files if check_multiple is True.

    Raises:
    FileNotFoundError: If the specified folder or file is not found.
    """
    if base_directory is None:
        base_directory = os.getcwd()  # Use current working directory if no base is provided
    
    for root, dirs, files in os.walk(base_directory):
        if check_multiple and not is_folder:
            # Search for multiple files by extension or name
            if search_by_extension:
                filtered_files = [os.path.join(root, f) for f in files if f.endswith(item)]
            else:
                filtered_files = [os.path.join(root, f) for f in files if f == item]
            
            if filtered_files:
                # Found files, return and stop searching further
                return filtered_files

        elif is_folder and item in dirs:
            # Return the first found directory if not checking for multiples
            return os.path.join(root, item)

        elif not is_folder and item in files:
            # Return the first found file if not checking for multiples
            return os.path.join(root, item)

    raise FileNotFoundError(f"No matching {item} found within the specified directory.")

def organize_file_into_folder(file_name):
    """
    Extracts the base name from a file, creates a folder named after the file without its extension,
    and moves the file into the newly created folder.
    
    :param file_name: str, the full path to the file
    """
    # Extract the directory path and base name without the extension
    folder_path = os.path.dirname(file_name)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    new_folder_path = os.path.join(folder_path, base_name)

    # Create a folder named after the file if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Move the file into the newly created folder
    new_file_path = os.path.join(new_folder_path, os.path.basename(file_name))
    shutil.move(file_name, new_file_path)
    print(f"Moved {file_name} to {new_file_path}")

    return new_file_path


def process_and_save_rois(tif_file, rois):
    """
    Processes a given TIFF file to extract and save regions of interest (ROIs) into a new directory.
    
    Args:
    tif_file (str): Path to the TIFF file.
    rois (list of tuples): A list of tuples, each representing an ROI in the format (x, y, width, height).
    """
    # Extract the base folder path from the TIFF file path
    base_folder = os.path.dirname(tif_file)

    # Define the new folder for saving ROIs, nested inside the base folder
    output_folder = os.path.join(base_folder, 'ROIs')

    # Open the original image
    image = Image.open(tif_file)
    
    # Get the base name for the image without the path and extension
    base_name = os.path.splitext(os.path.basename(tif_file))[0]
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each ROI
    for i, (x, y, width, height) in enumerate(rois):
        # Crop the image using the coordinates
        cropped_image = image.crop((x, y, width, height))
        
        # Format file name for the cropped image
        roi_file_name = f"{base_name}_roi_{x}_{y}_{width}_{height}.tif"
        full_path = os.path.join(output_folder, roi_file_name)
        
        # Save the cropped image as a TIFF file
        cropped_image.save(full_path, 'TIFF')

    print(f"All ROIs have been saved in {output_folder}")

    return output_folder

def save_csv_file(folder, df, file_name):
    """
    Save a CSV file with the specified name in the given folder.
    
    Args:
    folder (str): The path to the folder where the file will be saved.
    file_name (str): The name of the CSV file to save.
    
    Returns:
    str: The full path to the saved CSV file.
    """
    file_path = os.path.join(folder, file_name)
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")
    return file_path