from collections import defaultdict
from dotenv import load_dotenv
import os
import time
from urllib.parse import quote_plus
import uuid
import streamlit as st

import pandas as pd
from pymongo import ASCENDING, MongoClient

from Analysis.STORM.Models.molecule import Molecule
from Analysis.STORM.Models.track import Track

load_dotenv()


class STORMDatabaseManager:
    """Handles database connections and operations for the STORM database."""

    def __init__(self):
        self.username = os.getenv("STORM_DB_USER")
        self.password = os.getenv("STORM_DB_PASS")
        self.host = os.getenv("STORM_DB_HOST", "localhost")
        self.port = int(os.getenv("STORM_DB_PORT", 27017))
        self.database_name = os.getenv("STORM_DB_NAME", "storm_db")

        if not self.username or not self.password:
            raise ValueError("Database credentials not found in environment variables.")

        uri = f"mongodb://{self.username}:{quote_plus(self.password)}@{self.host}:{self.port}"
        self.client = MongoClient(uri)
        self.db = self.client[self.database_name]
        self.initialize_database()

    def initialize_database(self):
        """Initialize collections and indexes for the STORM database."""
        # Create collections if they do not exist and define indexes
        self.experiments = self.db['experiments']
        self.molecules = self.db['molecules']
        self.localizations = self.db['localizations']

        # Indexes for Experiments collection
        self.experiments.create_index([("date", ASCENDING)])
        self.experiments.create_index([("file_name", ASCENDING)])

        # Indexes for Molecules collection
        self.molecules.create_index([("experiment_id", ASCENDING)])

        # Indexes for Localizations collection
        self.localizations.create_index([("track_id", ASCENDING)])


    def save_metadata(self, metadata, file_name, folder_path):
        """
        Save experiment and extracted metadata to the MongoDB database.

        Args:
            metadata (list of dicts): List of metadata dictionaries with keys: 'id', 'type', 'value', 'tag'.
            file_name (str): The name of the file associated with the metadata.
            folder_path (str): The relative folder path where the experiment is stored.
        """
        # Generate a unique UUID for the experiment ID
        experiment_id = uuid.uuid4().hex
        # Extract 'Date' from metadata
        date_value = next((entry['value'] for entry in metadata if entry['tag'] == 'Date'), "Unknown")

        # Prepare the experiment document with embedded metadata
        experiment_document = {
            "_id": experiment_id,
            "file_name": file_name,
            "folder_path": folder_path,
            "date": date_value,
            "metadata": [
                {
                    "tag": item['tag'],
                    "name": item.get('id', ''),  # Adjusted for the key used in MongoDB document
                    "type": item['type'],
                    "value": item['value']
                }
                for item in metadata
            ]
        }

        # Insert the experiment document into the 'experiments' collection and return the inserted_id
        result = self.experiments.insert_one(experiment_document)
        return result.inserted_id

    def load_storm_metadata(self):
        """
        Load distinct metadata from the MongoDB database where `tag='PulseSTORM'`.

        Returns:
            dict: Metadata dictionary where keys are metadata names and values are lists of tuples (value, type).
        """
        pipeline = [
            {"$unwind": "$metadata"},  # Unwind the metadata array to process each item separately
            {"$match": {"metadata.tag": "pulsestorm"}},  # Match documents where metadata tag is 'PulseSTORM'
            {"$group": {  # Group to collect unique metadata
                "_id": {"name": "$metadata.name", "value": "$metadata.value", "type": "$metadata.type"},
                "uniqueValues": {"$addToSet": {"value": "$metadata.value", "type": "$metadata.type"}}
            }},
            {"$sort": {"_id.name": 1, "_id.value": 1}}  # Sort results by name and value
        ]

        results = self.experiments.aggregate(pipeline)

        # Organize the results into the desired dictionary format
        database_metadata = {}
        for result in results:
            name = result['_id']['name']
            entries = result['uniqueValues']
            if name not in database_metadata:
                database_metadata[name] = []
            for entry in entries:
                database_metadata[name].append((entry['value'], entry['type']))

        return database_metadata

    def storm_folders_without_localizations(self):
        """
        Retrieves experiments where TIFF files exist but have no associated molecule data,
        indicating that there are no localizations linked to these experiments.

        Returns:
            list: A list of unique (experiment_id, folder_path) dictionaries where no localizations exist.
        """
        pipeline = [
            {
                "$match": {
                    "time_series": {"$exists": False}  # Check for experiments where 'time_series' field does not exist or is empty
                }
            },
            {
                "$project": {
                    "experiment_id": "$_id",  # Project the _id as experiment_id
                    "folder_path": 1,  # Include folder_path in the output
                    "_id": 0  # Exclude the default _id field
                }
            }
        ]

        results = self.experiments.aggregate(pipeline)
        return list(results)  # Convert cursor to list

    

    def save_molecules(self, molecules, experiment_id):
        # Prepare data for molecules and localizations
        molecule_documents = []
        localization_documents = []

        for molecule in molecules:
            molecule_dict = molecule.to_dict()  # Ensure localizations are not included
            molecule_dict['experiment_id'] = experiment_id
            molecule_dict['tracks'] = []
            for track in molecule.tracks:
                # Append track without localizations
                track_dict = track.to_dict(embed_localizations=False)  # Do not embed localizations
                molecule_dict['tracks'].append(track_dict)

                for loc in track.localizations:
                    loc_dict = loc.to_dict()
                    loc_dict['track_id'] = track.track_id
                    localization_documents.append(loc_dict)
            
            molecule_documents.append(molecule_dict)

        # Insert molecules with embedded tracks (without localizations)
        if molecule_documents:
            self.molecules.insert_many(molecule_documents)
        # Insert localizations separately
        if localization_documents:
            self.localizations.insert_many(localization_documents)
            

    def save_time_series(self, time_series_df, experiment_id):
        """
        Saves time series data to the MongoDB database by appending it to the existing experiment document.

        Args:
            time_series_df (pd.DataFrame): DataFrame containing time-series metrics.
            experiment_id (ObjectId or str): ID referencing the experiment record.
        """
        # Convert DataFrame to a list of dictionaries (more suitable for MongoDB updates)
        time_series_data = time_series_df.to_dict('records')  # Convert entire DataFrame to a list of dictionaries

        # Update the experiment document by appending each time series record to the 'time_series' array
        for time_series_entry in time_series_data:
            self.experiments.update_one(
                {"_id": experiment_id},
                {"$push": {"time_series": time_series_entry}}
            )


    def get_experiment_settings(self, experiment_id):
        """
        Retrieves total frames and exposure time from experiment metadata where the tag is 'czi-pulsestorm'.

        Args:
            experiment_id (ObjectId or str): The experiment ID to query.

        Returns:
            tuple: (total_frames, exposure_time) or (None, None) if not found.
        """
        pipeline = [
            {"$match": {"_id": experiment_id}},  # Match the experiment by ID
            {"$unwind": "$metadata"},  # Unwind the metadata array to process each item separately
            {"$match": {
                "metadata.tag": "czi-pulsestorm",
                "metadata.name": {"$in": ["Frames", "Exposure"]}  # Match specific metadata entries
            }},
            {"$project": {  # Project to format the output
                "name": "$metadata.name",
                "value": "$metadata.value"
            }}
        ]

        results = list(self.experiments.aggregate(pipeline))
        metadata_values = {result['name']: float(result['value']) for result in results if isinstance(result['value'], (int, float))}

        total_frames = int(metadata_values.get('Frames', 0)) if 'Frames' in metadata_values else None
        exposure_time = int(metadata_values.get('Exposure', 0)) if 'Exposure' in metadata_values else None

        return total_frames, exposure_time

    
    
    def get_metadata(self, filters=None):
        """
        Retrieve metadata from the MongoDB database.

        Args:
            filters (dict, optional): Column names as keys and list of values as filters.

        Returns:
            dict: A dictionary where each experiment_id has its metadata assigned as {name: value}.
        """
        pipeline = [
            {"$match": {"time_series": {"$exists": True, "$ne": []}}}  # Only consider experiments with time_series data
        ]
    

        # Applying dynamic filters based on input
        if filters:
            for name, values in filters.items():
                pipeline.append({"$match": {
                    f"metadata.name": name,
                    f"metadata.value": {"$in": values}
                }})

        # Unwind metadata for processing
        pipeline.append({"$unwind": "$metadata"})

        # Group to assemble metadata per experiment
        pipeline.append({
            "$group": {
                "_id": "$_id",
                "folder_path": {"$first": "$folder_path"},
                "metadata": {
                    "$push": {
                        "name": "$metadata.name",
                        "value": "$metadata.value",
                        "type": "$metadata.type"
                    }
                }
            }
        })

        results = list(self.experiments.aggregate(pipeline))

        # Dictionary to store results
        metadata_dict = {}
        for result in results:
            exp_id = str(result["_id"])
            metadata_dict[exp_id] = {"Experiment": result["folder_path"]}

            for meta in result["metadata"]:
                prop_name = meta["name"]
                prop_type = meta["type"]
                prop_value = meta["value"]

                # Convert value based on type
                if prop_type == 'int':
                    converted_value = int(prop_value)
                elif prop_type == 'float':
                    converted_value = float(prop_value)
                elif prop_type == 'bool':
                    converted_value = prop_value.lower() in ('true', '1', 't')
                else:
                    converted_value = str(prop_value)  # Default to string

                metadata_dict[exp_id][prop_name] = converted_value

        return metadata_dict
    
    def get_grouped_molecules_and_tracks(_self, _experiment_ids):
        """
        Retrieves and groups molecules with their associated tracks for the given experiment IDs.
        
        Args:
            experiment_ids (list): List of experiment IDs.
        
        Returns:
            dict: A dictionary with experiment IDs as keys and lists of Molecule objects as values.
        """
        # Fetch all molecules for the specified experiment IDs
        query_result = list(_self.molecules.find({"experiment_id": {"$in": _experiment_ids}}))
        total_documents = len(query_result)
        
        if total_documents == 0:
            return {}
        
        grouped_molecules = defaultdict(list)
        
        # Setup for progress bar
        progress_bar = st.progress(0)
        start_time = time.time()
        update_interval = 500  # Update progress every 500 molecules
        step_log = st.empty()

        # Processing molecules
        for index, doc in enumerate(query_result):
            if (index + 1) % update_interval == 0 or index + 1 == total_documents:
                # Update the progress bar and log
                current_time = time.time()
                elapsed_time = current_time - start_time
                remaining_docs = total_documents - index - 1
                percent_complete = (index + 1) / total_documents
                progress_bar.progress(percent_complete)
                estimated_total_time = elapsed_time / (index + 1) * total_documents
                estimated_remaining_time = estimated_total_time - elapsed_time
                step_log.write(f"Processed {index + 1}/{total_documents} molecules in {elapsed_time:.2f} seconds. "
                            f"Estimated time remaining: {estimated_remaining_time / 60:.2f} minutes.")
            
            exp_id = doc["experiment_id"]
            # Track deduplication set
            seen_track_ids = set()

            # Unique tracks only
            tracks = []
            for track in doc.get("tracks", []):
                track_id = track.get("id")
                if track_id in seen_track_ids:
                    continue  # Skip duplicate
                seen_track_ids.add(track_id)

                track_obj = Track(
                    track_id=track_id,
                    start_frame=track.get("start_frame"),
                    end_frame=track.get("end_frame"),
                    intensity=track.get("intensity"),
                    offset=track.get("offset"),
                    bkgstd=track.get("bkgstd"),
                    uncertainty=track.get("uncertainty"),
                    on_time=track.get("on_time"),
                    off_time=track.get("off_time"),
                    x=track.get("x"),
                    y=track.get("y"),
                    molecule_id=doc["_id"]
                )
                tracks.append(track_obj)
            
            molecule = Molecule(
                molecule_id=doc["_id"],
                experiment_id=exp_id,
                start_track=doc.get("start_track"),
                end_track=doc.get("end_track"),
                total_on_time=doc.get("total_on_time"),
                num_tracks=len(tracks),
                tracks=tracks
            )
            
            grouped_molecules[exp_id].append(molecule)
        
        # Final update
        progress_bar.progress(1.0)
        total_time = time.time() - start_time
        step_log.write(f"Completed obtaining {total_documents} molecules in {total_time:.2f} seconds.")
        progress_bar.empty()  # Remove the progress bar widget after completion

        return grouped_molecules


    
    def get_grouped_time_series(self, experiment_ids):
        """
        Retrieves and groups time-series data for given experiment IDs.

        Args:
            experiment_ids (list): List of experiment IDs.

        Returns:
            dict: {experiment_id: DataFrame of time-series values}
        """
        if not experiment_ids:
            return {}

        # Use MongoDB's aggregation framework to filter and project time series data
        pipeline = [
            {"$match": {
                "_id": {"$in": experiment_ids}
            }},
            {"$project": {
                "time_series": 1,
                "_id": 1
            }}
        ]

        query_result = self.experiments.aggregate(pipeline)
        time_series_dict = {}

        # Iterate through the aggregation results
        for doc in query_result:
            experiment_id = doc["_id"]
            # If time_series data exists, convert it to DataFrame, otherwise create an empty DataFrame
            if 'time_series' in doc:
                df = pd.DataFrame(doc['time_series'])
            else:
                df = pd.DataFrame()
            
            time_series_dict[experiment_id] = df

        return time_series_dict
    

    def get_localizations_by_tracks(self, track_ids: list[str]) -> pd.DataFrame:
        """
        Retrieves localizations associated with a list of track IDs and returns them as a DataFrame.

        Args:
            track_ids (list of str): List of track IDs to fetch localizations for.

        Returns:
            pd.DataFrame: DataFrame containing localizations for the given tracks.
        """
        query = {"track_id": {"$in": track_ids}}
        cursor = self.localizations.find(query)

        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))

        if not df.empty:
            # Normalize field names to match your previous code
            df.rename(columns={
                "track_id": "TRACK_ID",
                "frame": "FRAME",
                "intensity": "INTENSITY"
            }, inplace=True)

        return df


if __name__ == "__main__":
    # Initialize STORM database manager
    storm_db = STORMDatabaseManager()

    # Step 1: Get all experiment IDs with molecules
    experiment_ids = storm_db.molecules.distinct("experiment_id")

    # Step 2: Fetch grouped molecules and their tracks
    grouped_molecules = storm_db.get_grouped_molecules_and_tracks(experiment_ids)

    # Step 3: Display info for first 10 molecules (across all experiments)
    count = 0
    for exp_id, molecules in grouped_molecules.items():
        for molecule in molecules:
            print(f"\n🔬 Molecule ID: {molecule.molecule_id}")
            print(f"📁 Experiment ID: {molecule.experiment_id}")
            print(f"📊 Total Tracks: {len(molecule.tracks)}")
            print(f"⏱️ Total On-Time: {molecule.total_on_time:.2f} frames")
            print(f"🧬 Track IDs: {[track.track_id for track in molecule.tracks]}")

            for track in molecule.tracks:
                print(f"  ▶ Track {track.track_id}: Start {track.start_frame}, End {track.end_frame}, Intensity {track.intensity}")
            
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break
