import json
import os
import shutil
from tkinter import filedialog
import tkinter as tk
from Analysis.SOAC.analytics_soac_filaments import soac_analytics_pipeline
from Analysis.SOAC.preprocessing_image_selection import *
from Analysis.SOAC.analytics_ridge_filaments import analyze_data
from Analysis.SOAC.soac_api import soac_api
from Analysis.STORM.track_storm import analyze_data_storm
from Analysis.STORM.molecule_merging import process_tracks
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
                exclude_files={'ridge_metrics.csv', 'soac_results.csv'},
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
        folder_path = filedialog.askdirectory(title="Select Folder for Blinking Statistics TrackMate")
        if folder_path:
            valid_folders = find_valid_folders(
                folder_path,
                required_files={'.czi','trackmate.csv'},
                exclude_files={'locs_blink_stats.csv', 'track_blink_stats.csv', 'mol_blink_stats.csv'},
            )

            print(valid_folders)

            for folder in valid_folders:
                
                czi_files = find_items(base_directory=folder, item='.czi', is_folder=False, check_multiple=True, search_by_extension=True)

                for czi_file in czi_files:
                    # Obtain file name without directory and extension
                    czi_filename = os.path.basename(czi_file).split(".c")[0]
                    tm_file = find_items(base_directory=folder, item=f'{czi_filename}_trackmate.csv', is_folder=False, check_multiple=False, search_by_extension=False)

                    # Add both files to a list
                    #tm_files = [czi_file, trackmate_file]

                    #if len(czi_files) > 1:
                    #    tm_file = organize_file_into_folder(file_name = czi_file, files = tm_files)

                    #tm_file = organize_file_into_folder(file_name = czi_file, files = tm_files)

                    new_folder = os.path.dirname(tm_file)
                    df = pd.read_csv(tm_file)

                    localizations, tracks, molecules = process_tracks(df, 'trackmate')

                    save_csv_file(new_folder, localizations, f'{os.path.basename(tm_file).split(".c")[0]}_locs_blink_stats.csv')
                    save_csv_file(new_folder, tracks, f'{os.path.basename(tm_file).split(".c")[0]}_track_blink_stats.csv')
                    save_csv_file(new_folder, molecules, f'{os.path.basename(tm_file).split(".c")[0]}_mol_blink_stats.csv')

    else:
        print("Invalid option selected.")

    root.destroy()  # Close the Tkinter instance

if __name__ == "__main__":
    main()
