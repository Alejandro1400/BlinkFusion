import os
import tkinter as tk
from tkinter import filedialog
from Data_access.metadata_manager import read_tiff_metadata
from Data_access.storm_db import STORMDatabaseManager


def extract_and_save_metadata(db_manager, tif_file_path):
    """Extracts metadata and saves it to the database."""
    filename = os.path.basename(tif_file_path)

    # Extract metadata for both tags
    pulsestorm_metadata = read_tiff_metadata(tif_file_path, 'pulsestorm')
    czi_pulsestorm_metadata = read_tiff_metadata(tif_file_path, 'czi-pulsestorm')

    # Add tag field to each entry before saving
    for entry in pulsestorm_metadata:
        entry['tag'] = 'pulsestorm'
    
    for entry in czi_pulsestorm_metadata:
        entry['tag'] = 'czi-pulsestorm'

    # Combine metadata from both tags
    all_metadata = pulsestorm_metadata + czi_pulsestorm_metadata

    # Save to database
    db_manager.save_metadata(all_metadata, filename)

    print(f"Metadata for {filename} saved successfully.")


def process_all_tif_files(db_manager, folder_path):
    """Recursively searches for all .tif files in the given folder and processes them."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                tif_file_path = os.path.join(root, file)
                print(f"Processing: {tif_file_path}")
                extract_and_save_metadata(db_manager, tif_file_path)


def main():
    """Main function to handle folder selection and metadata extraction for all .tif files."""
    # Initialize Tkinter to select folder
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    folder_path = filedialog.askdirectory(title="Select the main folder containing TIFF files")

    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print(f"Selected folder: {folder_path}")

    # Initialize STORM database
    storm_db = STORMDatabaseManager(folder_path)

    # Process all .tif files in the folder (recursively)
    process_all_tif_files(storm_db, folder_path)

    # Close database connection
    storm_db.close()
    print("Process completed.")


if __name__ == "__main__":
    main()
