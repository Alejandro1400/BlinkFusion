import os
from tkinter import filedialog
import tkinter as tk
from Analysis.analytics_ridge_pipeline import analyze_data
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
                # Proceed with further operations using the `box_client` if needed
                pass
    elif user_choice == '2':
        print("\nFile Explorer selected.")
        print("1: Process Data")
        print("2: Use Dashboard")
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

    else:
        print("Invalid option selected.")

    root.destroy()  # Close the Tkinter instance

if __name__ == "__main__":
    main()
