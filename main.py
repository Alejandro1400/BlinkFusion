import json
import os
import shutil
from tkinter import filedialog
import tkinter as tk
from Analysis.SOAC.analytics_soac_filaments import soac_analytics_pipeline
from Analysis.SOAC.preprocessing_image_selection import *
from Analysis.SOAC.analytics_ridge_filaments import analyze_data
from Analysis.SOAC.soac_api import soac_api
from Analysis.STORM.analytics_storm_pipeline import analyze_data_storm
from Data_access import box_connection
from Data_access.file_explorer import *

def main():

    # Hide the main window of Tkinter
    root = tk.Tk()
    root.withdraw()

    # User choices via command line
    print("Select your option:")
    print("1: SOAC Biopolymerous Filamentous Network Analysis")
    print("2: STORM Microscopy Data Analysis")
    user_choice = input("Type '1', '2': ")

    if user_choice == '1':
        folder_path = filedialog.askdirectory(title="Select Folder for SOAC Analysis")
        if folder_path:
            config_path = find_items(item='ridge_detector_param.json', is_folder=False)
            parameter_file = find_items(item='batch_parameters.txt', is_folder=False)
            executable_path = find_items(item='batch_soax_v3.7.0.exe', is_folder=False)

            valid_folders = find_valid_folders(
                folder_path,
                required_files={'.tif'},
                exclude_files={'snakes.csv', 'junctions.csv'},
                exclude_folders={'ROIs'}
            )

            print(valid_folders)

            for folder in valid_folders:
                
                # Find tif files in folder
                tif_files = find_items(base_directory=folder, item='.tif', is_folder=False, check_multiple=True, search_by_extension=True)
                    
                for tif_file in tif_files: 
                    
                    if len(tif_files) > 1:
                        tif_file = organize_file_into_folder(tif_file)

                    folder = os.path.dirname(tif_file)

                    # Preprocess the image and select ROIs
                    ROIs, metrics = preprocessing_image_selection(tif_file, config_path, num_ROIs=16)

                    save_csv_file(folder, metrics, f'{os.path.basename(tif_file).split(".")[0]}_ridge_metrics.csv')

                    output_folder = process_and_save_rois(tif_file, ROIs)
                    print(f"Data processed for: {folder}")

                    snakes, junctions = soac_api(output_folder, parameter_file, executable_path, folder)

                    snakes = soac_analytics_pipeline(snakes, junctions)

                    # The file name will be {tif_file name}_soac_results
                    save_csv_file(folder, snakes, f'{os.path.basename(tif_file).split(".")[0]}_soac_results.csv')


    elif user_choice == '2':
        folder_path = filedialog.askdirectory(title="Select Folder for SOAC Processing")
        if folder_path:
            valid_folders = [f for f in os.listdir(folder_path) if check_data(
                os.path.join(folder_path, f),
                required_files={'.tif'},
                exclude_files={'snakes.csv', 'junctions.csv'}
            )]
            for folder in valid_folders:
                results, junctions, image = processing_data(folder)
                processed = analyze_data(results, junctions, image)  
                save_processed_data(processed, folder_path, folder) 
                print(f"Data processed for: {folder}")
        else:
            print("No folder selected.")




            
        folder_path = filedialog.askdirectory(title="Select Folder for Processing")
        if folder_path:
            valid_folders = [f for f in os.listdir(folder_path) if check_data(
                os.path.join(folder_path, f),
                required_files={'Results.csv', 'Junctions.csv', '.tif'},
                exclude_files={'Processed.csv'}
            )]
            for folder in valid_folders:
                results, junctions, image = processing_data(folder)
                processed = analyze_data(results, junctions, image)  
                save_processed_data(processed, folder_path, folder) 
                print(f"Data processed for: {folder}")
        else:
            print("No folder selected.")
    elif user_choice == '2':
        folder_path = filedialog.askdirectory(title="Select Folder for Dashboard")
        if folder_path:
            #valid_folders = folders_for_dashboard(folder_path)
            for folder in valid_folders:
                processed = dashboard_data(folder)
                print(f"Dashboard data ready for: {folder}")
        else:
            print("No folder selected.")

    elif user_choice == '3':
        file_path = filedialog.askopenfilename(title="Select File for Processing", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            processed_storm = analyze_data_storm(file_path)
            print("Data processed.")
        else:
            print("No file selected.")

        


    else:
        print("Invalid option selected.")

    root.destroy()  # Close the Tkinter instance

if __name__ == "__main__":
    main()
