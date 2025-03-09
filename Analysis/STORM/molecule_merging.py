import time
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from Analysis.STORM.Models.molecule import Molecule
from Analysis.STORM.localization_tracker import LocalizationTracker, merge_localizations
from Analysis.STORM.preprocessing import Preprocessor


class MoleculeTracker:
    """
    Processes tracking events from localized molecules and merges them into molecular representations.
    """

    def __init__(self, df: pd.DataFrame, file_type: str, max_distance=100):
        """
        Initializes the molecule tracker.

        Parameters:
        - df (pd.DataFrame): DataFrame containing localizations or tracking data.
        - file_type (str): Type of file ('trackmate' or 'thunderstorm') to determine processing steps.
        """
        self.df = df
        self.file_type = file_type
        self.tracking_events = None
        self.max_distance = max_distance
        self.molecules = []
        self.molecule_id_counter = 1

                    
    def merge_tracking_events(self):
        """
        Cluster tracking events that are close to each other within a maximum distance.
        
        Updates:
        - Assigns `molecule_id` to each track, grouping tracks into molecules.
        """
        # Create a KD-tree for tracking events
        track_positions = np.array([(track.x, track.y) for track in self.tracks])
        kd_tree = KDTree(track_positions)

        # Iterate through tracks to assign molecule IDs
        for track in self.tracks:
            if track.molecule_id != -1:
                continue

            track.molecule_id = self.molecule_id_counter
            frames_set = set(range(track.start_frame, track.end_frame + 1)) - track.gaps

            # Query KDTree for nearby tracks
            indices = kd_tree.query_ball_point((track.x, track.y), r=self.max_distance)

            
            for idx in indices:
                nearby_track = self.tracks[idx]
                
                if nearby_track.molecule_id != -1:
                    continue  # Skip already assigned tracks

                # Obtain frames 
                new_frames_set = set(range(nearby_track.start_frame, nearby_track.end_frame + 1)) - nearby_track.gaps

                # Check for frame overlap
                if not frames_set.intersection(new_frames_set):
                    nearby_track.molecule_id = self.molecule_id_counter
                    frames_set.update(new_frames_set)
            
            # Increment molecule ID for the next unique molecule
            self.molecule_id_counter += 1


    def update_merged_locs(self):
        """
        Update the tracking events list to reflect merged tracking events and remove obsolete tracks from tracking_events.
        """

        track_changes = {}

        # Group by molecule ID
        molecule_groups = {}
        for track in self.tracks:
            if track.molecule_id not in molecule_groups:
                molecule_groups[track.molecule_id] = []
            molecule_groups[track.molecule_id].append(track)

        for molecule_id, group in molecule_groups.items():
            sorted_group = sorted(group, key=lambda t: t.start_frame)

            for i in range(len(sorted_group) - 1):
                if sorted_group[i].track_id in track_changes:
                    continue  # Skip already merged tracks

                for j in range(i + 1, len(sorted_group)):
                    track_ini = sorted_group[i]
                    track_fin = sorted_group[j]

                    # Check if they are consecutive
                    if track_fin.start_frame - track_ini.end_frame <= 2:
                        track_changes[track_fin.track_id] = track_ini.track_id

                        # Merge frames
                        frames_ini = set(range(track_ini.start_frame, track_ini.end_frame + 1)) - track_ini.gaps
                        frames_fin = set(range(track_fin.start_frame, track_fin.end_frame + 1)) - track_fin.gaps
                        frames_union = frames_ini.union(frames_fin)

                        # Update gaps
                        track_ini.gaps.update(track_fin.gaps)
                        track_ini.gaps.difference_update(frames_union)

                        # Update track properties
                        track_ini.start_frame = min(track_ini.start_frame, track_fin.start_frame)
                        track_ini.end_frame = max(track_ini.end_frame, track_fin.end_frame)
                        track_ini.intensity += track_fin.intensity
                        track_ini.offset = (track_ini.offset + track_fin.offset) / 2
                        track_ini.bkgstd += track_fin.bkgstd
                        track_ini.uncertainty = (track_ini.uncertainty + track_fin.uncertainty) / 2

                        # Merge localizations
                        track_ini.localizations.extend(track_fin.localizations)

                        # Remove merged track
                        self.tracks.remove(track_fin)
                    else:
                        break  # Stop checking if the tracks are no longer consecutive


    def track_blinking_times(self):
        """
        Adds ON_TIME and OFF_TIME for tracking events, considering only tracks within the same molecule.
        """
        # Calculate ON_TIME for each tracking event
        for track in self.tracking_events:
            track.on_time = track.end_frame - track.start_frame + 1
            track.off_time = 0  # Default to 0, will be updated below

        # Group tracking_events by molecule ID
        molecule_groups = {}
        for track in self.tracking_events:
            if track.molecule_id not in molecule_groups:
                molecule_groups[track.molecule_id] = []
            molecule_groups[track.molecule_id].append(track)

        # Process each molecule group
        for molecule_id, group in molecule_groups.items():
            sorted_group = sorted(group, key=lambda t: t.start_frame)

            # Calculate OFF_TIME for each track except the last one
            for i in range(len(sorted_group) - 1):
                track_ini = sorted_group[i]
                track_next = sorted_group[i + 1]
                track_ini.off_time = track_next.start_frame - track_ini.end_frame


    def create_molecules(self):
        """
        Create Molecule objects from tracking events.
        """
        molecule_dict = {}

        for track in self.tracks:
            if track.molecule_id not in molecule_dict:
                molecule_dict[track.molecule_id] = Molecule(track.molecule_id)
            molecule_dict[track.molecule_id].add_track(track)

        self.molecules = list(molecule_dict.values())


    def process_tracks(self):
        """ Process the entire file and merge tracking events. """

        if self.file_type == 'trackmate':
            # Prepare the columns
            preprocessor = Preprocessor(self.df, self.file_type)
            self.raw_localizations = preprocessor.prepare_columns()
            self.tracking_events = preprocessor.create_tracking_events()
        elif self.file_type == 'thunderstorm':
            locstracker = LocalizationTracker()
            self.tracking_events = locstracker.merge_localizations(self.df)

        # Merge molecules
        self.merge_tracking_events()

        # Update localizations
        self.update_merged_locs()

        # Calculate blinking times
        self.track_blinking_times()

        # Create molecules
        self.create_molecules()
        
        # Determine photobleaching
        #molecules = bleaching_identification(molecules, blink_tracking_events)

        return self.molecules

