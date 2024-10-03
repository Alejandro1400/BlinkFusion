import numpy as np
from ridge_detection.lineDetector import LineDetector 
from ridge_detection.params import Params, load_json
from ridge_detection.helper import displayContours
from PIL import Image
from mrcfile import open as mrcfile_open


def prepare_image(image_path):
    # Open the image as a multi frame tiff file
    img = Image.open(image_path)

    # Check if the image is a multi-frame tiff file
    if hasattr(img, 'n_frames') and img.n_frames > 1:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img))  # Convert the current frame to a numpy array after seeking
        img = np.mean(frames, axis=0)  # Calculate the average intensity across all frames
    else:
        img = np.array(img)  # Convert single frame image to numpy array

    # Convert the numpy array to an image
    img = Image.fromarray(img)

    # Print the image mean intensity, area, min and max intensity
    print(f"Image Mean Intensity: {np.mean(np.array(img))}")
    print(f"Image Area: {img.size}")
    print(f"Image Min Intensity: {np.min(np.array(img))}")
    print(f"Image Max Intensity: {np.max(np.array(img))}")
    
    
    # Convert image to 8-bit if it is 16-bit
    if img.mode != 'I':
        img = img.convert('I')

        array = np.uint8(np.array(img) / 256)
        img = Image.fromarray(array)

    # Print the image's intensity mean and standard deviation
    print(f"Image Mean Intensity: {np.mean(np.array(img))}")
    print(f"Image Standard Deviation: {np.std(np.array(img))}")

    return img



def select_ROIs(img, num_ROIs=None, ROI_size=None):

    # Get image dimensions
    width, height = img.size

    # Initialize list to hold ROI coordinates
    ROIs = []

    if num_ROIs is not None:
        # Calculate the size of each ROI assuming a square grid layout
        grid_size = int(np.ceil(np.sqrt(num_ROIs)))  # Determine the grid size needed
        ROI_width, ROI_height = width // grid_size, height // grid_size

        # Generate ROI coordinates based on the grid
        for i in range(grid_size):
            for j in range(grid_size):
                left = i * ROI_width
                top = j * ROI_height
                right = left + ROI_width
                bottom = top + ROI_height
                if right <= width and bottom <= height:
                    ROIs.append((left, top, right, bottom))

    elif ROI_size is not None:
        # Calculate the number of ROIs based on the provided ROI size
        ROI_width, ROI_height = ROI_size

        # Generate ROI coordinates based on the fixed size
        for i in range(0, width, ROI_width):
            for j in range(0, height, ROI_height):
                right = i + ROI_width
                bottom = j + ROI_height
                if right <= width and bottom <= height:
                    ROIs.append((i, j, right, bottom))

    # Placeholder to use the config file for additional settings (if needed)
    # Process config settings here

    # Return the list of ROIs
    return ROIs


def ridge_detection_params(img, config, line_width=3):
    # Obtain the mean pixel intensity value of the image
    image_array = np.array(img)
    mean_intensity = np.mean(np.array(img))
    #Obtain the standard deviation of the pixel intensity values of the image
    std_intensity = np.std(np.array(img))

    # Lower contrast is the mean intensity and high contrast is the mean plus std deviation
    lower_contrast = mean_intensity
    higher_contrast = mean_intensity + std_intensity

    # Calculate sigma, lower threshold, and upper threshold
    # Calculate sigma from line width
    sigma = line_width / (2 * np.sqrt(3)) + 0.5

    # Calculate upper threshold
    Tu = (0.17 * higher_contrast * 2 * (line_width / 2) * np.exp(- (line_width / 2) ** 2 / (2 * sigma ** 2))) / np.sqrt(2 * np.pi * sigma ** 3)

    # Calculate lower threshold
    Tl = (0.17 * lower_contrast * 2 * (line_width / 2) * np.exp(- (line_width / 2) ** 2 / (2 * sigma ** 2))) / np.sqrt(2 * np.pi * sigma ** 3)

    # Update the parameters in the config dictionary
    config['mandatory_parameters']['Sigma'] = sigma
    config['mandatory_parameters']['Lower_Threshold'] = Tl
    config['mandatory_parameters']['Upper_Threshold'] = Tu

    # Updae the optional parameters in the config dictionary
    config['optional_parameters']['Line_width'] = line_width
    config['optional_parameters']['Low_contrast'] = lower_contrast
    config['optional_parameters']['High_contrast'] = higher_contrast

    return config



def ridges_statistics(ridges, junctions):
    # Calculate number of ridges
    num_ridges = len(ridges)
    
    # Calculate total length of each ridge and average length
    total_length = 0
    for ridge in ridges:
        x_coords = ridge.col
        y_coords = ridge.row
        # Calculate the length of the ridge using the Euclidean distance between points
        length = sum(np.sqrt((x_coords[i] - x_coords[i - 1]) ** 2 + (y_coords[i] - y_coords[i - 1]) ** 2) 
                     for i in range(1, len(x_coords)))
        total_length += length

    average_length = total_length / num_ridges if num_ridges > 0 else 0

    # Count the number of junctions
    num_junctions = len(junctions)

    return num_ridges, average_length, num_junctions


def detect_ridges(img, config):
    # Calculate parameters from image
    config = ridge_detection_params(img, config)

    # Initialize the line detector
    detect = LineDetector(params=config)
    
    # Perform line detection
    result = detect.detectLines(img)
    resultJunction = detect.junctions

    # Ridge statistics
    num_ridges, avg_length, num_junctions = ridges_statistics(result, resultJunction)

    # Return the detection results
    return num_ridges, avg_length, num_junctions


def preprocessing_image_selection(image_path, config_file, num_ROIs=None, ROI_size=None):
    # Load the configuration
    config = load_json(config_file)

    # Prepare the image
    image = prepare_image(image_path)

    # Select ROIs from the image
    ROIs = select_ROIs(image, num_ROIs=num_ROIs, ROI_size=ROI_size)

    # Initialize list to hold detection results for each ROI
    detection_results = []

    # Process each ROI
    for roi in ROIs:
        # Crop the image to the ROI
        img = image.crop(roi)
        num_ridges, avg_length, num_junctions = detect_ridges(img, config)
        print(f"ROI: {roi}, Num Ridges: {num_ridges}, Avg Length: {avg_length}, Num Junctions: {num_junctions}")

        if num_ridges/num_junctions > 1.5:
            # Display the original cropped image with a title saying the region 
            img.show(title=f"ROI: {roi}")

    return detection_results