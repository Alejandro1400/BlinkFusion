import os
import numpy as np
from ridge_detection.lineDetector import LineDetector 
from ridge_detection.params import Params, load_json
from ridge_detection.helper import displayContours
from PIL import Image
from mrcfile import open as mrcfile_open
from tabulate import tabulate


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
    
    # Convert image to 8-bit if it is 16-bit
    if img.mode != 'I':
        img = img.convert('I')

    img_array = np.array(img) 

    # Normalize the image to 8-bit
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255
    img = Image.fromarray(img_array.astype(np.uint8))

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
    # Calculate number of ridges and junctions
    num_ridges = len(ridges)
    num_junctions = len(junctions)
    # Calculate Ridge/Junction Ratio
    ridge_junction_ratio = num_ridges / num_junctions if num_junctions > 0 else 0
    
    # Calculate total length of each ridge and average length
    total_length = 0
    average_widths = []
    mean_intensities = []
    for ridge in ridges:
        x_coords = ridge.col
        y_coords = ridge.row
        # Calculate the length of the ridge using the Euclidean distance between points
        length = sum(np.sqrt((x_coords[i] - x_coords[i - 1]) ** 2 + (y_coords[i] - y_coords[i - 1]) ** 2) 
                     for i in range(1, len(x_coords)))
        total_length += length

        avg_width = np.mean(ridge.width_l + ridge.width_r)
        average_widths.append(avg_width)

        mean_intensity = np.mean(ridge.intensity)
        mean_intensities.append(mean_intensity)
    
    # Calculate mean length and coefficient of variation
    mean_length = total_length / num_ridges if num_ridges > 0 else 0
    cv_length = np.std(average_widths) / np.mean(average_widths) if np.mean(average_widths) != 0 else 0
    mean_intensity = np.mean(mean_intensities)
    cv_width = np.std(average_widths) / np.mean(average_widths) if np.mean(average_widths) != 0 else 0
    
    # Normalize the values using Min-Max normalization (0-1 scaling)
    metrics = np.array([num_ridges, ridge_junction_ratio, mean_length, cv_length, mean_intensity, cv_width])
    #metric_names = ["Number of Ridges", "Ridge/Junction Ratio", "Mean Length", "CV Length", "Mean Intensity", "CV Width"]

    # Printing metrics with names
    #print("Metrics: ", " ".join([f"{name}: {metric:.2f}" for name, metric in zip(metric_names, metrics)]))

    return metrics


def detect_ridges(img, config):
    # Calculate parameters from image
    config = ridge_detection_params(img, config)

    # Initialize the line detector
    detect = LineDetector(params=config)
    
    # Perform line detection
    result = detect.detectLines(img)
    resultJunction = detect.junctions

    # Ridge statistics
    metrics = ridges_statistics(result, resultJunction)
    
    # Return the detection results
    return metrics


def preprocessing_image_selection(image_path, config_file, scaling_method='min_max', num_ROIs=None, ROI_size=None):
    # Load the configuration
    config = load_json(config_file)

    # Prepare the image
    image = prepare_image(image_path)

    #image.show()
    # Save
    #image.save("original_image.tiff")

    # Select ROIs from the image
    ROIs = select_ROIs(image, num_ROIs=num_ROIs, ROI_size=ROI_size)

    # Initialize list to hold detection results for each ROI
    metrics_results = []

    # Process each ROI
    for roi in ROIs:
        # Crop the image to the ROI
        img = image.crop(roi)
        roi_metrics = detect_ridges(img, config)
        metrics_results.append(roi_metrics)

    # Now that all metrics are collected, we can normalize across all ROIs
    # Stack all ROI metrics for vectorized operations
    all_metrics = np.array(metrics_results)
    
    # Names and initial weights for the metrics
    metric_names = ["Number of Ridges", "Ridge/Junction Ratio", "Mean Length", "CV Length", "Mean Intensity", "CV Width"]
    weights = np.array([35, 35, 10, 10, 5, 5]) / 100.0

    min_vals = np.min(all_metrics, axis=0)
    max_vals = np.max(all_metrics, axis=0)
    scaled_metrics = (all_metrics - min_vals) / (max_vals - min_vals)
    for i in range(all_metrics.shape[1]):
        if metric_names[i].startswith('CV'):  # Invert scaling for 'CV' metrics
            scaled_metrics[:, i] = 1-scaled_metrics[:, i]

    # Calculate ROI Quality for each ROI
    roi_qualities = np.dot(scaled_metrics, weights)

    # Filter out ROIs with quality < 0.4
    high_quality_indices = roi_qualities >= 0.4
    filtered_ROIs = np.array(ROIs)[high_quality_indices]
    filtered_metrics = all_metrics[high_quality_indices]
    filtered_scaled_metrics = scaled_metrics[high_quality_indices]
    filtered_roi_qualities = roi_qualities[high_quality_indices]

    # Print the results in a table
    headers = ["Metric", "Real Value", "Scaled Value"]
    print("Min max scaling")
    for roi, quality in zip(filtered_ROIs, filtered_roi_qualities):
        print(f"ROI: {roi}, Quality: {quality:.2f}")
        print("\n")

    # Create a mask image with a black background (or any other color)
    #masked_image = Image.new('I', image.size, 0)

    # Copy only high-quality ROIs to the masked image
    #for roi in filtered_ROIs:
    #    crop_area = image.crop(roi)
    #    masked_image.paste(crop_area, roi)

    # Save or show the resulting image
    #masked_image.show()
    #masked_image.save("high_quality_ROIs_min_max.tiff")


    # Set new weights for the metrics
    weights = np.array([35, 20, 25, 10, 5, 5]) / 100.0

    # Standard scale the metrics
    scaled_metrics = (filtered_metrics - np.mean(filtered_metrics, axis=0)) / np.std(filtered_metrics, axis=0)

    # Calculate ROI Quality for each ROI
    roi_qualities = np.dot(scaled_metrics, weights)

    # Filter out ROIs with negative quality
    high_quality_indices = roi_qualities >= 0
    filtered_ROIs = filtered_ROIs[high_quality_indices]
    filtered_metrics = filtered_metrics[high_quality_indices]
    filtered_scaled_metrics = scaled_metrics[high_quality_indices]
    filtered_roi_qualities = roi_qualities[high_quality_indices]

    # Print the results in a table
    print("Standard scaling")
    for roi, filtered_metric, filtered_scaled_metric, quality in zip(filtered_ROIs, filtered_metrics, filtered_scaled_metrics, filtered_roi_qualities):
        print(f"ROI: {roi}, Quality: {quality:.2f}")
        print("Metrics: ")
        print(filtered_metric)
        print ("Scaled Metrics: ")
        print(filtered_scaled_metric)
        print("\n")

    # Create a mask image with a black background (or any other color)
    #masked_image = Image.new('I', image.size, 0)

    # Copy only high-quality ROIs to the masked image
    #for roi in filtered_ROIs:
    #    crop_area = image.crop(roi)
    #    masked_image.paste(crop_area, roi)

    # Save or show the resulting image
    #masked_image.show()
    #masked_image.save("high_quality_ROIs.tiff")

    #Filter out top 3 ROIs
    top_ROIs = filtered_ROIs[np.argsort(filtered_roi_qualities)[::-1][:3]]

    # Create a mask image with a black background (or any other color)
    #masked_image = Image.new('I', image.size, 0)

    # Copy only top 3 ROIs to the masked image
    #for roi in top_ROIs:
    #    crop_area = image.crop(roi)
    #    masked_image.paste(crop_area, roi)

    # Save or show the resulting image
    #masked_image.show()
    # Save the image
    #masked_image.save("top_3_ROIs.tiff")

    return top_ROIs

from PIL import Image

def save_rois_image(image_path, rois, output_path):
    """
    Crops regions of interest from an image and saves each as a new TIFF file.

    Args:
    image_path (str): Path to the original image.
    rois (list of tuples): A list of tuples, each containing the coordinates (x, y, width, height) of a rectangle ROI.
    output_path (str): Directory where the cropped images will be saved.
    """
    # Open the original image
    image = Image.open(image_path)
    
    # Get the base name for the image without the path and extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Process each ROI
    for i, (x, y, width, height) in enumerate(rois):
        # Crop the image using the coordinates
        cropped_image = image.crop((x, y, x + width, y + height))
        
        # Create a file name that includes the original image name and ROI coordinates
        roi_file_name = f"{base_name}_roi_{i}_{x}_{y}_{width}_{height}.tif"
        
        # Create the full path to save the cropped image
        full_path = os.path.join(output_path, roi_file_name)
        
        # Save the cropped image as a TIFF file
        cropped_image.save(full_path)

    print(f"All ROIs have been saved in {output_path}")


    

    



    