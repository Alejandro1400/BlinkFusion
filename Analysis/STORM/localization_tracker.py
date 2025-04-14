import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import time
import streamlit as st

from Analysis.STORM.Models.localization import Localization
from Analysis.STORM.Models.track import Track

class LocalizationTracker:
    def __init__(self, min_frames: int = 3, max_gaps: int = 2, max_distance: float = 200):
        """
        Initializes the localization tracker with configurable tracking parameters.
        
        Parameters:
        - min_frames (int): Minimum number of frames required for a track to be considered valid.
        - max_gaps (int): Maximum allowed gaps (missing frames) before a track is finalized.
        - max_distance (float): Maximum allowed distance for associating localizations to an existing track.
        """
        self.min_frames = min_frames
        self.max_gaps = max_gaps
        self.max_distance = max_distance
        self.active_tracks = []
        self.completed_tracks = []
        self.track_id_counter = 1
        self.complete_track_id_counter = 1
        self.time_log = {}
        self.last_frame = 0
        self.actual_frame = 0
        self.locs_merged = 0
        self.assigned_tracks = 0
        self.start_time = time.time()


    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names and data types in the given DataFrame.
        
        Parameters:
        - df (pd.DataFrame): Input DataFrame containing localization data.
        
        Returns:
        - pd.DataFrame: Processed DataFrame with standardized column names and data types.
        """
        df.columns = [col.upper() for col in df.columns]
        column_mapping = {
            "X [NM]": "X", "Y [NM]": "Y", "SIGMA [NM]": "SIGMA",
            "INTENSITY [PHOTON]": "INTENSITY", "OFFSET [PHOTON]": "OFFSET",
            "BKGSTD [PHOTON]": "BKGSTD", "UNCERTAINTY_XY [NM]": "UNCERTAINTY",
            "UNCERTAINTY [NM]": "UNCERTAINTY"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        if "ID" not in df.columns or df["ID"].isna().all():
            df["ID"] = range(1, len(df) + 1)
        
        dtype_mapping = {
            "ID": int, "FRAME": int, "X": float, "Y": float, "SIGMA": float,
            "INTENSITY": float, "OFFSET": float, "BKGSTD": float, "UNCERTAINTY": float
        }
        df = df.astype(dtype_mapping)
        return df


    def process_frame_data(self, frame_data: pd.DataFrame) -> list:
        """
        Converts DataFrame rows into a list of Localization objects.
        
        Parameters:
        - frame_data (pd.DataFrame): DataFrame containing localization data for a single frame.
        
        Returns:
        - list: List of Localization objects.
        """
        return [
            Localization(row.ID, row.FRAME, row.X, row.Y, row.SIGMA, 
                        row.INTENSITY, row.OFFSET, row.BKGSTD, row.UNCERTAINTY)
            for _, row in frame_data.iterrows()
        ]


    def process_active_tracks(self, frame_localizations: list, kd_tree: KDTree, frame: int) -> set:
        """
        Matches localizations to existing tracks using a KDTree for fast nearest-neighbor lookup.
        
        Parameters:
        - frame_localizations (list): List of Localization objects for the current frame.
        - kd_tree (KDTree): KDTree built from the localizations for efficient distance computation.
        - frame (int): The current frame being processed.
        
        Returns:
        - set: Indices of localizations that were assigned to existing tracks.
        """
        assigned_indices = set()
        tracks_to_remove = []

        for track in self.active_tracks:
            dist_last, nearest_idx_last = kd_tree.query((track.localizations[-1].x, track.localizations[-1].y), distance_upper_bound=self.max_distance)
            dist_weighted, nearest_idx_weighted = kd_tree.query((track.x, track.y), distance_upper_bound=self.max_distance)

            nearest_idx = nearest_idx_last if dist_last < dist_weighted else nearest_idx_weighted
            dist = min(dist_last, dist_weighted)

            if dist < self.max_distance and nearest_idx not in assigned_indices:
                assigned_indices.add(nearest_idx)
                matched_loc = frame_localizations[nearest_idx]
                track.add_localization(matched_loc)  # Automatically updates weighted position
                track.gaps_counter = 0  # Reset gap count
            else:
                track.gaps_counter += 1

                if track.gaps_counter >= self.max_gaps:
                    if len(track.localizations) >= self.min_frames:
                        track.start_frame = track.localizations[0].frame
                        track.end_frame = track.localizations[-1].frame
                        self.completed_tracks.append(track)

                        self.complete_track_id_counter += 1
                        self.assigned_tracks +=1
                        self.locs_merged += len(track.localizations)

                    tracks_to_remove.append(track)

                track.gaps.add(frame)

        for track in tracks_to_remove:
            self.active_tracks.remove(track)

        return assigned_indices


    def merge_localizations(self, df: pd.DataFrame, step_log) -> list:
        """
        Processes localization data frame by frame, tracking movement and forming trajectories.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing localization data for multiple frames.
        
        Returns:
        - list: List of completed Track objects.
        """
        df = self.standardize_columns(df)
        df['TRACK_ID'] = 0
        df = df.sort_values('FRAME')

        localization_progress = st.progress(0, text="Merging localizations...")
        total_frames = df['FRAME'].max()

        # Step 3: Process frames
        for frame, frame_data in df.groupby('FRAME'):
            # Check if the frame is mod 500
            if frame != self.actual_frame and frame % 500 == 0:
                elapsed_time = time.time() - self.start_time
                self.start_time = time.time()
                self.last_frame = self.actual_frame
                self.actual_frame = frame
                frame_range = f"{self.last_frame} to {self.actual_frame}"
                if frame != 0:
                    step_log.write(f"🔹 Frame {frame_range}: Localizations: {self.locs_merged}, Tracks: {self.assigned_tracks}, Time: {elapsed_time:.2f} sec")

                localization_progress.progress(frame / total_frames, text=f"Processing frame {frame}/{total_frames}")

                self.time_log[frame_range] = elapsed_time
                self.locs_merged = 0
                self.assigned_tracks = 0

            frame_localizations = self.process_frame_data(frame_data)
            kd_tree = KDTree([(loc.x, loc.y) for loc in frame_localizations])

            assigned_indices = self.process_active_tracks(frame_localizations, kd_tree, frame)

            for i, loc in enumerate(frame_localizations):
                if i not in assigned_indices:
                    new_track = Track()
                    new_track.add_localization(loc)
                    new_track.start_frame = frame
                    new_track.end_frame = frame
                    self.active_tracks.append(new_track)
                    self.track_id_counter += 1
        
        localization_progress.empty()  # Clear progress bar
        return self.completed_tracks



