import os
import re
import shutil
from xml.dom import minidom
import pandas as pd
from skimage import io
import tifffile
from Analysis.SOAC.analytics_ridge_filaments import analyze_data
from PIL import Image, TiffImagePlugin
import xml.etree.ElementTree as ET

def check_data(folder_path, required_files, exclude_files=None, exclude_folders=None, single_file=False):
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


def find_items(base_directory=None, item=None, is_folder=False, search_by_extension=False, check_multiple=False):
    """
    Find the specified item within the base directory.

    Args:
    base_directory (str): The base directory to search within.
    item (str): The item to search for (file name or extension).
    is_folder (bool): Whether the item is a folder (default is False).
    search_by_extension (bool): Whether to search by extension (default is False).
    check_multiple (bool): Whether to check for multiple items (default is False).

    Returns:
    str or list: The path to the found item or a list of paths if check_multiple is True.
    """
    if base_directory is None:
        base_directory = os.getcwd()  # Use current working directory if no base is provided
    
    for root, dirs, files in os.walk(base_directory):
        # If searching for files
        if not is_folder:
            # If checking for multiple files
            if check_multiple:
                if search_by_extension:
                    filtered_files = [os.path.join(root, f) for f in files if f.endswith(item)]
                else:
                    filtered_files = [os.path.join(root, f) for f in files if f == item]
                
                if filtered_files:
                    return filtered_files  # Return all found files
            else:
                # If search by extension is True and we're not checking for multiple files
                if search_by_extension:
                    for f in files:
                        if f.endswith(item):
                            return os.path.join(root, f)  # Return the first file that matches the extension
                else:
                    if item in files:
                        return os.path.join(root, item)  # Return the first exact match file

        # If searching for a folder
        elif is_folder and item in dirs:
            return os.path.join(root, item)

    raise FileNotFoundError(f"No matching {item} found within the specified directory.")


def organize_file_into_folder(file_name, files=None):
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

    if files == None:
        # Move the file into the newly created folder
        new_file_path = os.path.join(new_folder_path, os.path.basename(file_name))
        shutil.move(file_name, new_file_path)
        print(f"Moved {file_name} to {new_file_path}")
    else:
        for file in files:
            new_file_path = os.path.join(new_folder_path, os.path.basename(file))
            shutil.move(file, new_file_path)
            print(f"Moved {file} to {new_file_path}")

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

def read_folder_structure(file_path):
    """
    Read the folder structure from a text file and return the contents as a list of folder types.
    
    Args:
    file_path (str): The path to the text file containing the folder structure.
    
    Returns:
    list: A list of folder types based on the file content.
    """
    folder_types = []
    
    # Open the text file and read each line
    with open(file_path, 'r') as file:
        folder_types = [line.strip() for line in file if line.strip()]
    
    return folder_types

def assign_structure_folders(base_directory, folder_structure_file, data_folders):
    """
    Assign the data folders to their respective structure folders based on the folder structure file.
    
    Args:
    base_directory (str): The base directory to search for data folders.
    folder_structure_file (str): The path to the text file containing the folder structure.
    data_folders (list): A list of full data folder paths to assign to structure folders.
    
    Returns:
    df: A dataframe with folder_structure as columns and data_folders as rows with assigned folders.
    """
    # Read the folder structure from the text file
    folder_structure = read_folder_structure(folder_structure_file)
    
    # Initialize an empty list to store the folder mappings
    folder_mappings = {folder_type: [] for folder_type in folder_structure}
    folder_mappings['identifier'] = []
    
    # Iterate through each data folder (full path)
    for folder in data_folders:
        # Get the relative path from the base directory
        relative_path = os.path.relpath(folder, base_directory)
        
        # Split the relative path by the directory separator
        folder_names = relative_path.split(os.sep)
        
        # Assign each folder name to the corresponding structure based on the folder structure order
        for i, folder_type in enumerate(folder_structure):
            if i < len(folder_names):
                folder_mappings[folder_type].append(folder_names[i])
            else:
                folder_mappings[folder_type].append(None)  # Handle missing folders if the structure is deeper than the actual data

        folder_mappings['identifier'].append(relative_path)

    # Convert all titles to uppercase
    folder_mappings = {key.upper(): value for key, value in folder_mappings.items()}
    
    # Create a dataframe from the folder mappings
    df = pd.DataFrame(folder_mappings)
    
    return df


def read_tiff_metadata(tif_file_path, root_tag='prop', id_filter=None):
    """
    Read specific metadata from a TIFF file's ImageDescription as a plain string based on given root tag and optional ID filter.
    
    Args:
    tif_file_path (str): Path to the TIFF file.
    root_tag (str): The XML root tag to search for (e.g., 'prop', 'custom-prop').
    id_filter (str, optional): Specific ID to search for within the root tags.

    Returns:
    dict: Dictionary of found metadata values corresponding to specified IDs.
    """
    with tifffile.TiffFile(tif_file_path) as tif:
        image_description = tif.pages[0].tags['ImageDescription'].value

    found_metadata = []
    # Define a regex to extract tag attributes correctly handling spaces
    attr_pattern = re.compile(r'(\w+)="([^"]*)"')

    # Process each line in the image description
    for line in image_description.split('\n'):
        if f'<{root_tag}' in line:
            # Parse the attributes using regex
            attrs = dict(attr_pattern.findall(line))

            if id_filter is None or attrs.get('id', '').startswith(id_filter):
                prop_id = attrs.get('id')
                prop_type = attrs.get('type')
                prop_value = attrs.get('value')

                # Convert value to the appropriate type based on 'type' attribute
                if prop_type == 'int':
                    prop_value = int(prop_value)
                elif prop_type == 'float':
                    prop_value = float(prop_value)
                elif prop_type == 'bool':
                    prop_value = prop_value.lower() in ('true', '1', 't')
                    
                metadata = {'id': prop_id, 'type': prop_type, 'value': prop_value}
                found_metadata.append(metadata)

                if id_filter and prop_id == id_filter:
                    return metadata

    return found_metadata


def append_metadata_tags(tif_file_path, new_tif_file_path, tags):
    """
    Append metadata tags directly to the ImageDescription of a TIFF file.

    Args:
    tif_file_path (str): Path to the original TIFF file.
    new_tif_file_path (str): Path where the modified TIFF file will be saved.
    tags (list of dicts): List of metadata tags to append, each dict containing 'id', 'type', and 'value'.
    """
    with tifffile.TiffFile(tif_file_path) as tif:
        image = tif.asarray()
        metadata = tif.pages[0].tags.get('ImageDescription', '').value if 'ImageDescription' in tif.pages[0].tags else ''

        # Find the position to insert new tags (just before </MetaData>)
        insert_pos = metadata.rfind('</MetaData>')
        if insert_pos == -1:
            # If </MetaData> not found, append at the end of the existing metadata
            insert_pos = len(metadata)
            metadata += '<MetaData>'

        # Prepare new tags as a string to insert
        new_tags_str = ''
        for tag in tags:
            new_tag = f'<{tag["root_tag"]} id="{tag["id"]}" type="{tag["type"]}" value="{tag["value"]}"/>\n'
            new_tags_str += new_tag

        # Insert new tags into the metadata
        updated_metadata = metadata[:insert_pos] + new_tags_str + metadata[insert_pos:]

        # Save the modified image to a new file with updated metadata
        tifffile.imsave(new_tif_file_path, image, description=updated_metadata)


def parse_metadata_input(metadata_input):
    """
    Parse the metadata input from the Streamlit text input into a list of tuples.
    The input should be in the format: "Name, Type, Value & Name, Type, Value"
    """
    tags = []
    entries = metadata_input.split('&')
    for entry in entries:
        parts = entry.strip().split(',')
        if len(parts) == 3:
            tag_name, value_type, value = parts
            tag_name = tag_name.strip()
            value = value.strip()
            if value_type.strip().lower() == 'int':
                value = int(value)
            elif value_type.strip().lower() == 'float':
                value = float(value)
            elif value_type.strip().lower() == 'bool':
                value = value.lower() in ('true', '1', 't')
            tags.append((tag_name, value, value_type.strip().lower()))
    return tags


def get_open_laser_intensity(metadata):
    """
    Find the laser that is 'Open' and return its ID along with the corresponding intensity,
    assuming metadata is given in the format {'id': prop_id, 'type': prop_type, 'value': prop_value}.
    
    Args:
    metadata (list of dicts): List containing metadata with laser statuses and intensities.
    
    Returns:
    str, float: Laser ID and its intensity if open, otherwise None, None.
    """
    # Create dictionaries to hold pairs of intensity and power for easy lookup
    intensity_dict = {}
    power_dict = {}
    
    # Categorize entries into intensity or power based on the id
    for data in metadata:
        if "Intensity" in data['id']:
            intensity_dict[data['id']] = data['value']
        elif "Power" in data['id']:
            power_dict[data['id']] = data['value']

    # Search for an open laser and its corresponding intensity
    for power_key, power_value in power_dict.items():
        if power_value == "Open":  # Assuming the value directly tells if it's open
            print(f"Found open laser: {power_key}")
            # Extract the numeric identifier for the laser from the power key
            laser_number = re.findall(r'\d+', power_key)
            laser_number = f"Laser {laser_number[0]}" if laser_number else None
            if laser_number:
                # Construct the corresponding intensity key
                match_intensity_key = next((key for key in intensity_dict.keys() if laser_number in key), None)
                if match_intensity_key:
                    intensity = intensity_dict[match_intensity_key]
                    print(f"Matched intensity key: {match_intensity_key}")
                    laser_id = match_intensity_key.split('(')[-1].split(')')[0]
                    return laser_id, float(intensity)  # Convert intensity to float
    return None, None

def extract_gain_value(description):
    # Use a regular expression to find the "Multiplication Gain" value
    match = re.search(r'Multiplication Gain: (\d+)&#', description)
    if match:
        # Return the value found, converting it to an integer
        return int(match.group(1))
    else:
        # Return None if no match is found
        return None


def aggregate_metadata_info(metadata_dict):
    """
    Aggregate metadata info from all files into a single DataFrame with unique values listed,
    converting lists of values to a comma-separated string to ensure compatibility.
    """
    metadata_info = {}

    for file_metadata in metadata_dict.values():
        for data in file_metadata:
            value = data['value']
            if value != 'N/A':  # Skip 'N/A' values
                if data['id'] not in metadata_info:
                    metadata_info[data['id']] = {'count': 1, 'values': {value}}
                else:
                    metadata_info[data['id']]['count'] += 1
                    metadata_info[data['id']]['values'].add(value)

    # Prepare data for DataFrame
    data_for_df = []
    for id, info in metadata_info.items():
        data_for_df.append({
            'ID': id, 
            'Count': info['count'], 
            'Unique Values': ', '.join(map(str, sorted(info['values'])))
        })

    # Create DataFrame from the prepared data
    df = pd.DataFrame(data_for_df)
    df.set_index('ID', inplace=True)

    return df


def process_tiff_metadata(file_path):
    """
    Process a TIFF file to extract, format, and return specific metadata as a list of dictionaries.

    Args:
    file_path (str): Path to the TIFF file.

    Returns:
    list of dicts: A list containing dictionaries of formatted metadata.
    """
    # Helper function to safely read metadata with a default
    def safe_read_metadata(file, root_tag, id_filter):
        data = read_tiff_metadata(file, root_tag=root_tag, id_filter=id_filter)
        return data['value'] if data and 'value' in data else None

    # Extract specific metadata entries
    date = safe_read_metadata(file_path, 'prop', 'acquisition-time-local')
    pixel_size_x = safe_read_metadata(file_path, 'prop', 'pixel-size-x')
    pixel_size_y = safe_read_metadata(file_path, 'prop', 'pixel-size-y')
    dimension = f"{pixel_size_x} x {pixel_size_y}" if pixel_size_x and pixel_size_y else None

    spatial_calibration_x = safe_read_metadata(file_path, 'prop', 'spatial-calibration-x')
    spatial_calibration_y = safe_read_metadata(file_path, 'prop', 'spatial-calibration-y')
    spatial_calibration_unit = safe_read_metadata(file_path, 'prop', 'spatial-calibration-units')

    description = safe_read_metadata(file_path, 'prop', 'Description')
    gain = extract_gain_value(description) if description else None

    lasers = read_tiff_metadata(file_path, 'custom-prop', 'ALC Laser')
    laser_id, intensity = get_open_laser_intensity(lasers) if lasers else (None, None)

    # Format metadata into a dictionary
    tif_metadata = [
        {'root_tag': 'tif-pulsestorm', 'id': 'Date', 'type': 'string', 'value': date.split(' ')[0] if date else None},
        {'root_tag': 'tif-pulsestorm', 'id': 'Pixel Dimensions', 'type': 'string', 'value': dimension},
        {'root_tag': 'tif-pulsestorm', 'id': 'Laser', 'type': 'string', 'value': laser_id},
        {'root_tag': 'tif-pulsestorm', 'id': 'Laser Intensity', 'type': 'float', 'value': intensity},
        {'root_tag': 'tif-pulsestorm', 'id': 'Gain', 'type': 'float', 'value': gain},
        {'root_tag': 'tif-pulsestorm', 'id': 'Pixel Size X', 'type': 'float', 'value': spatial_calibration_x},
        {'root_tag': 'tif-pulsestorm', 'id': 'Pixel Size Y', 'type': 'float', 'value': spatial_calibration_y},
        {'root_tag': 'tif-pulsestorm', 'id': 'Pixel Size Units', 'type': 'string', 'value': spatial_calibration_unit}
    ]

    return [entry for entry in tif_metadata if entry['value'] is not None]

