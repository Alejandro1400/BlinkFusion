import os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from skimage import io
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tabulate import tabulate
import tkinter as tk

# Load data function
# Input: folder_path (str) - path to the folder containing the data files
# Output: data (dict) - dictionary containing the data from the JSON file
#         image (ndarray) - NumPy array containing the image data
# It searches for a tif and json file inside the folder_path and loads the data
def load_data(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            with open(os.path.join(folder_path, file), 'r') as json_file:
                data = json.load(json_file)
        elif file.endswith('.tif'):
            image = io.imread(os.path.join(folder_path, file))
    return data, image

# Function to extract filament metrics
# Input: data (dict) - dictionary containing the data from the JSON file
#        image (ndarray) - NumPy array containing the image data
# Output: lengths (list) - list of filament lengths
#         sinuosities (list) - list of filament sinuosities
#         filament_intensities (list) - list of average filament intensities
def extract_metrics(data, image):
    lengths = [filament['length'] for filament in data['filaments']]
    sinuosities = [filament['sinuosity'] for filament in data['filaments'] if not np.isnan(filament['sinuosity'])]
    # Ignore values in sinusoite that are higher than 2
    sinuosities = [sinuosity for sinuosity in sinuosities if sinuosity < 2]
    filament_intensities = [np.mean([image[int(y), int(x)] for x, y in zip(filament['x'], filament['y'])]) for filament in data['filaments']]
    # Normalize the intensity values to 0-1 range
    # Convert the list to a NumPy array for efficient computation
    intensities_array = np.array(filament_intensities)

    # Normalize the intensities
    min_intensity = np.min(intensities_array)
    max_intensity = np.max(intensities_array)
    filament_intensities_norm = (intensities_array - min_intensity) / (max_intensity - min_intensity)
    filament_intensities_norm = list(filament_intensities_norm)
    return lengths, sinuosities, filament_intensities, filament_intensities_norm

# Function to perform basic statistical analysis
# Input: lengths (list) - list of filament lengths
#        sinuosities (list) - list of filament sinuosities
#        filament_intensities (list) - list of average filament intensities
#        image (ndarray) - NumPy array containing the image data
# Output: average_length (float) - average filament length
#         std_length (float) - standard deviation of filament lengths
#         snr (float) - signal-to-noise ratio
#         average_intensity (float) - average filament intensity
#         std_intensity (float) - standard deviation of filament intensities
#         average_sinuosity (float) - average filament sinuosity
#         spatial_density (float) - spatial density of filaments
def basic_analysis(data, lengths, sinuosities, filament_intensities, image):
    average_length = np.mean(lengths)
    std_length = np.std(lengths)
    signal_pixels = [image[int(y), int(x)] for filament in data['filaments'] for x, y in zip(filament['x'], filament['y'])]
    noise_pixels = [image[y, x] for x, y in zip(np.random.randint(0, image.shape[1], 1000), np.random.randint(0, image.shape[0], 1000)) if image[y, x] not in signal_pixels]
    signal_mean = np.mean(signal_pixels)
    noise_std = np.std(noise_pixels)
    snr = signal_mean / noise_std
    average_intensity = np.mean(filament_intensities)
    std_intensity = np.std(filament_intensities)
    average_sinuosity = np.mean(sinuosities)
    std_sinuosity = np.std(sinuosities)
    area = image.shape[0] * image.shape[1]
    spatial_density = sum(filament['length'] * filament['size'] for filament in data['filaments']) / area
    return average_length, std_length, average_intensity, std_intensity, average_sinuosity, std_sinuosity, spatial_density, snr

# Categories values function
# Input: values  (list) - list of values to categorize
#        num_bins (int) - number of bins to categorize the values into
# Output: categories (list) - list of tuples containing the range, count, and percentage of values in each category
def categorize_values(all_values, num_bins=16):
    # Determine global range across all values
    min_val = min(min(vals) for vals in all_values.values() if vals)
    max_val = max(max(vals) for vals in all_values.values() if vals)

    # Define bins based on the global range
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    # Prepare a dictionary to collect categorized data by folder
    categorized_data = {folder: {"Counts": [], "Percentages": []} for folder in all_values.keys()}
    bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]

    # Categorize values for each folder using the defined bins
    for folder, values in all_values.items():
        counts, _ = np.histogram(values, bins=bins)
        percentages = 100 * counts / np.sum(counts) if np.sum(counts) > 0 else [0] * len(counts)
        categorized_data[folder]["Counts"] = counts
        categorized_data[folder]["Percentages"] = percentages

    # Create a DataFrame for a more structured table
    rows = []
    for bin_label in bin_labels:
        row = {"Range": bin_label}
        for folder in all_values.keys():
            row[f"{folder} Count"] = categorized_data[folder]["Counts"][bin_labels.index(bin_label)]
            row[f"{folder} Percentage"] = categorized_data[folder]["Percentages"][bin_labels.index(bin_label)]
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison_boxplots(all_lengths, all_sinuosities, all_intensities):
    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Sort folders by mean values of length to determine the order
    sorted_folders = sorted(all_lengths.items(), key=lambda x: np.mean(x[1]))
    sorted_labels = [folder for folder, _ in sorted_folders]

    # Define a common function to prepare data and plot
    def prepare_and_plot(ax, data_dict, title):
        # Use the sorted order
        data = [data_dict[folder] for folder in sorted_labels if folder in data_dict]

        # Create the boxplot
        ax.boxplot(data, labels=sorted_labels, patch_artist=True, vert=True)
        ax.set_title(title)
        ax.set_xlabel('Folder')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)

    # Plot each metric
    prepare_and_plot(axes[0], all_lengths, 'Length')
    prepare_and_plot(axes[1], all_intensities, 'Intensity')
    prepare_and_plot(axes[2], all_sinuosities, 'Sinuosity')

    # Adjust layout and display the plot
    plt.tight_layout()  # Adjust the rect to fit the suptitle
    plt.show()

# Create combined plots function
# Input: lengths (list) - list of filament lengths
#        filament_intensities (list) - list of average filament intensities
#        sinuosities (list) - list of filament sinuosities
#        data (dict) - dictionary containing the data from the JSON file
# Output: None
def create_combined_plots_with_stats(folder, lengths, filament_intensities, sinuosities, data):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # Set the title of the figure
    fig.suptitle(f'Analysis for {folder}', fontsize=16)
    
    # Helper function to plot histograms with normal fit
    def plot_hist_with_stats(ax, values, title, xlabel, ylabel, bins, color='grey'):
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        count_val = len(values)
        
        # Plot histogram
        n, bins_out, patches = ax.hist(values, bins=bins, color=color, edgecolor='black', density=True)
        
        # Fit a normal distribution to the data
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_val, std_val)
        ax.plot(x, p, 'b', linewidth=2)
        
        # Set titles and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Legend with statistics
        legend_text = f'Mean: {mean_val:.2f}\nStdDev: {std_val:.2f}\nN: {count_val}'
        ax.legend(['Normal Fit', legend_text], title='Statistics', loc='upper right')

    # Histogram of Filament Lengths with custom binning
    plot_hist_with_stats(axs[0, 0], lengths, 'Histogram of Filament Lengths', 'Length', 'Frequency', bins=range(int(min(lengths)), int(max(lengths)) + 1))

    # Histogram of Average Filament Intensities with custom binning
    plot_hist_with_stats(axs[0, 1], filament_intensities, 'Histogram of Average Filament Intensities', 'Intensity', 'Frequency', bins=range(int(min(filament_intensities)), int(max(filament_intensities)) + 1, 25))

    # Histogram of Filament Sinuosity with custom binning
    plot_hist_with_stats(axs[1, 0], sinuosities, 'Histogram of Filament Sinuosity', 'Sinuosity', 'Frequency', bins=100)

    # Spatial Density Plot
    filament_positions = np.array([[x, y] for filament in data['filaments'] for x, y in zip(filament['x'], filament['y'])])
    scaler = StandardScaler()
    filament_positions_scaled = scaler.fit_transform(filament_positions)
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    dbscan.fit(filament_positions_scaled)
    labels = dbscan.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    percentages = 100 * counts / counts.sum()
    scatter = axs[1, 1].scatter(filament_positions[:, 0], filament_positions[:, 1], c=labels, cmap='viridis', s=2)
    axs[1, 1].invert_yaxis()
    #legend_labels = [f'Group {label}: {percentage:.2f}%' if label != -1 else f'Noise: {percentage:.2f}%' for label, percentage in zip(unique_labels, percentages)]
    #handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(label/max(unique_labels)), markersize=6) for label in unique_labels]
    #axs[1, 1].legend(handles, legend_labels, title="Cluster Labels", loc='upper right')
    axs[1, 1].set_title('Spatial Density of Filaments (pixels)')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


# Main analysis function
# Input: folder_path (str) - path to the folder containing
# Output: None
def analyze_data(folder_path, multiple_folders):
    results = []
    all_lengths, all_sinuosities, all_intensities, all_intensities_norm, all_data = {}, {}, {}, {}, {}

    if multiple_folders:
        for subdir, dirs, files in os.walk(folder_path):
            for dir in dirs:
                subfolder_path = os.path.join(subdir, dir)
                data, image = load_data(subfolder_path)
                lengths, sinuosities, filament_intensities, filament_intensities_norm = extract_metrics(data, image)
                stats = basic_analysis(data, lengths, sinuosities, filament_intensities_norm, image)
                results.append((dir, *stats))
                all_lengths[os.path.basename(subfolder_path)] = lengths
                all_sinuosities[os.path.basename(subfolder_path)] = sinuosities
                all_intensities[os.path.basename(subfolder_path)] = filament_intensities
                all_intensities_norm[os.path.basename(subfolder_path)] = filament_intensities_norm
                all_data[os.path.basename(subfolder_path)] = data
    else:
        data, image = load_data(folder_path)
        lengths, sinuosities, filament_intensities, filament_intensities_norm = extract_metrics(data, image)
        stats = basic_analysis(data, lengths, sinuosities, filament_intensities_norm, image)
        results.append((os.path.basename(folder_path), *stats))
        all_lengths[os.path.basename(folder_path)] = lengths
        all_sinuosities[os.path.basename(folder_path)] = sinuosities
        all_intensities[os.path.basename(folder_path)] = filament_intensities
        all_intensities_norm[os.path.basename(folder_path)] = filament_intensities_norm
        all_data[os.path.basename(folder_path)] = data
    
    # Print introduction to the metrics analysis
    print("------------------\nFILAMENT METRICS ANALYSIS\n------------------")
    print("The following metrics were calculated for the filaments in the dataset:\n")
    
    columns = ["Folder", "Mean Length", "Std Dev Length", "Average Intensity", "Std Dev Intensity", "Average Sinuosity", "Std Dev Sinuoisity", "Spatial Density", "SNR"]
    df = pd.DataFrame(results, columns=columns)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    # Categorize and display categories
    length_categories = categorize_values(all_lengths)
    sinuosity_categories = categorize_values(all_sinuosities)
    intensity_categories = categorize_values(all_intensities_norm)

    # Display the categorized data for each metric using tabulate
    print("\nLength Categories:")
    print(tabulate(length_categories, headers="keys", tablefmt="psql", showindex=False))

    print("\nSinuosity Categories:")
    print(tabulate(sinuosity_categories, headers="keys", tablefmt="psql", showindex=False))

    print("\nIntensity Categories:")
    print(tabulate(intensity_categories, headers="keys", tablefmt="psql", showindex=False))  

    # Plot comparison boxplots
    plot_comparison_boxplots(all_lengths, all_sinuosities, all_intensities_norm)
    
    # Create combined plots
    for folder, data in all_data.items():
        lengths = all_lengths[folder]
        sinuosities = all_sinuosities[folder]
        intensities = all_intensities[folder]
        create_combined_plots_with_stats(folder, lengths, intensities, sinuosities, data)



# Main
def main():
    # Are there multiple folders to analyze
    multiple_folders = True
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()

    analyze_data(folder_path, multiple_folders)

if __name__ == '__main__':
    main()





