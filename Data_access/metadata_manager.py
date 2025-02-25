from datetime import datetime
import os
import re
import czifile
import pandas as pd
import tifffile
import xml.etree.ElementTree as ET
import streamlit as st


def read_tiff_metadata(tif_file_path, root_tag='prop', id_filter=None):
    """
    Read specific metadata from a TIFF file's ImageDescription as a plain string based on given root tags and optional ID filter.
    
    Args:
    tif_file_path (str): Path to the TIFF file.
    root_tags (Union[str, list]): The XML root tag(s) to search for (e.g., 'prop', ['custom-prop', 'default-prop']).
    id_filter (str, optional): Specific ID to search for within the root tags.

    Returns:
    dict: Dictionary of found metadata values corresponding to specified IDs.
    """
    # Ensure root_tags is a list even if a single string is provided
    if isinstance(root_tag, str):
        root_tag = [root_tag]

    with tifffile.TiffFile(tif_file_path) as tif:
        image_description = tif.pages[0].tags['ImageDescription'].value

    found_metadata = []
    attr_pattern = re.compile(r'(\w+)="([^"]*)"')

    # Process each line in the image description
    for line in image_description.split('\n'):
        # Check if the line contains any of the specified root tags
        if any(f'<{tag}' in line for tag in root_tag):
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

                # If a specific ID filter is matched, return immediately
                if id_filter and prop_id == id_filter:
                    return metadata

    return found_metadata


def czi_2_tiff(czi_filepath, tif_folder, hierarchy_folders, tags):

    # Load CZI file
    with czifile.CziFile(czi_filepath) as czi:
        image_data = czi.asarray()
        metadata = czi.metadata()

    image_desc = '<MetaData>\n'
    czi_metadata = f'<czi-file-metadata id="Description" type="string" value="{metadata}">\n'

    ins_metadata = read_czi_metadata_from_string(metadata)
    tags += ins_metadata

    folder_path =[]
    for folder in hierarchy_folders:
        for data in tags:
            if folder in data['id']:
                folder_path.append(data['value'])

    folder_path.append(os.path.basename(czi_filepath).replace('.czi', ''))

    tif_filepath = os.path.join(tif_folder, *folder_path, os.path.basename(czi_filepath).replace('.czi', '.tif'))

    # Prepare new tags as a string to insert
    new_tags_str = ''
    for tag in tags:
        value_str = f'"{tag["value"]}"' 

        # Format the new tag string
        new_tag = f'<{tag["tag"]} id="{tag["id"]}" type="{tag["type"]}" value={value_str}/>\n'
        new_tags_str += new_tag

    image_desc_fin = '</MetaData>\n'

    updated_metadata = image_desc + czi_metadata + new_tags_str + image_desc_fin

    # Check if the folders for the new file path exist, and create them if not
    new_folder = os.path.dirname(tif_filepath)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Save the modified image to a new file with updated metadata
    tifffile.imsave(tif_filepath, image_data, description=updated_metadata)


def read_czi_metadata_from_string(metadata):
    # Parse XML metadata from a string
    root = ET.fromstring(metadata)
    
    # Extracting specific values from metadata
    creation_date = root.find('.//CreationDate').text if root.find('.//CreationDate') is not None else 'Not available'
    fw_fov_position = root.find('.//FWFOVPosition').text if root.find('.//FWFOVPosition') is not None else 'Not available'
    size_x = root.find('.//SizeX').text if root.find('.//SizeX') is not None else 'Not available'
    size_y = root.find('.//SizeY').text if root.find('.//SizeY') is not None else 'Not available'
    size_t = root.find('.//SizeT').text if root.find('.//SizeT') is not None else 'Not available'
    laser_name = root.find('.//LaserName').text if root.find('.//LaserName') is not None else 'Not available'

    creation_date = datetime.strptime(creation_date, '%Y-%m-%dT%H:%M:%S')
    date = creation_date.strftime('%Y%m%d')

    dimensions = f"{size_x} x {size_y}"

    frames = int(size_t)

    czi_metadata = [
        {'tag': 'czi-pulsestorm', 'id': 'Date', 'type': 'string', 'value': date if date else None},
        {'tag': 'czi-pulsestorm', 'id': 'Pixel Dimensions', 'type': 'string', 'value': dimensions if dimensions else None},
        {'tag': 'czi-pulsestorm', 'id': 'Laser', 'type': 'string', 'value': laser_name if laser_name else None},
        {'tag': 'czi-pulsestorm', 'id': 'Frames', 'type': 'int', 'value': frames if frames else None},
        {'tag': 'czi-pulsestorm', 'id': 'Mode', 'type': 'string', 'value': fw_fov_position if fw_fov_position else None}
    ]

    return czi_metadata


def extract_values_from_title(title):
    # Define a regex pattern to match key-value pairs
    pattern = r'(angle|laser|exp|gain)(\d+\.?\d*)'
    
    # Search the string for all occurrences of the pattern
    matches = re.findall(pattern, title)
    
    # Convert matches into a dictionary
    values = {key: float(value) for key, value in matches}

    pulsestorm_metadata = [
        {'tag': 'czi-pulsestorm', 'id': 'Angle', 'type': 'float', 'value': values.get('angle')},
        {'tag': 'czi-pulsestorm', 'id': 'Laser', 'type': 'float', 'value': values.get('laser')},
        {'tag': 'czi-pulsestorm', 'id': 'Exposure', 'type': 'float', 'value': values.get('exp')},
        {'tag': 'czi-pulsestorm', 'id': 'Gain', 'type': 'float', 'value': values.get('gain')}
    ]
    
    return pulsestorm_metadata


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
            value_str = f'"{tag["value"]}"'
    
            # Format the new tag string
            new_tag = f'<{tag["root_tag"]} id="{tag["id"]}" type="{tag["type"]}" value={value_str}/>\n'
            new_tags_str += new_tag

        # Insert new tags into the metadata
        updated_metadata = metadata[:insert_pos] + new_tags_str + metadata[insert_pos:]

        # Check if the folders for the new file path exist, and create them if not
        new_folder = os.path.dirname(new_tif_file_path)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        # Save the modified image to a new file with updated metadata
        tifffile.imsave(new_tif_file_path, image, description=updated_metadata)


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

@st.cache_data
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