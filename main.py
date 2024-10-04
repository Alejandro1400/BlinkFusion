import json
import os
from tkinter import filedialog
import tkinter as tk
from Analysis.SOAC.preprocessing_image_selection import *
from Analysis.SOAC.analytics_ridge_pipeline import analyze_data
from Analysis.STORM.analytics_storm_pipeline import analyze_data_storm
from Data_access import box_connection
from Data_access.file_explorer import *

def main():

    # Hide the main window of Tkinter
    root = tk.Tk()
    root.withdraw()

    # User choices via command line
    print("Select your option:")
    print("1: Box")
    print("2: File Explorer")
    user_choice = input("Type '1' to use Box or '2' to use File Explorer: ")

    if user_choice == '1':
            print("\nBox selected.")
            auth_file = find_item(item_name='auth_info.txt', is_folder=False)
            box_client = box_connection.box_signin(auth_file)

            if box_client:
                database_json = box_connection.database_stats(box_client, root_dir='Filament Data')
                # Print folder names from the JSON
                data = json.loads(database_json)
                print("\nFolder Names:")
                for date in data['dates']:
                    print(f"Date: {date['date_name']} Id: {date['date_id']}")
                    for sample in date['samples']:
                        print(f"  Sample: {sample['sample_name']} Id: {sample['sample_id']}")
                        for cell in sample['cells']:
                            print(f"    Cell: {cell['cell_name']} Id: {cell['cell_id']}")

            #box_connection.change_item_name(box_client, '281680114916', 'C1_', item_type='folder')









    elif user_choice == '2':

        print("\nFile Explorer selected.")
        print("1: Process Data SOAC")
        print("2: Use Dashboard")
        print("3: Process Data STORM")
        process_choice = input("Type '1' to process data or '2' to use dashboard: ")

        if process_choice == '1':
            folder_path = filedialog.askdirectory(title="Select Folder for Processing")
            if folder_path:
                valid_folders = folders_for_processing(folder_path)
                for folder in valid_folders:
                    results, junctions, image = processing_data(folder)
                    processed = analyze_data(results, junctions, image)  
                    save_processed_data(processed, folder_path, folder) 
                    print(f"Data processed for: {folder}")
            else:
                print("No folder selected.")
        elif process_choice == '2':
            folder_path = filedialog.askdirectory(title="Select Folder for Dashboard")
            if folder_path:
                valid_folders = folders_for_dashboard(folder_path)
                for folder in valid_folders:
                    processed = dashboard_data(folder)
                    print(f"Dashboard data ready for: {folder}")
            else:
                print("No folder selected.")

        elif process_choice == '3':
            file_path = filedialog.askopenfilename(title="Select File for Processing", filetypes=[("CSV Files", "*.csv")])
            if file_path:
                processed_storm = analyze_data_storm(file_path)
                print("Data processed.")
            else:
                print("No file selected.")



    elif user_choice == '3':
        file_path = filedialog.askopenfilename(title="Select File for Processing", filetypes=[("Tiff Files", "*.tiff *.tif")])
        if file_path:
            config_path = find_item(item_name='ridge_detector_param.json', is_folder=False)
            preprocessing_image_selection(file_path, config_path, num_ROIs=16)
        else:
            print("No file selected.")  


    else:
        print("Invalid option selected.")

    root.destroy()  # Close the Tkinter instance

if __name__ == "__main__":
    main()
