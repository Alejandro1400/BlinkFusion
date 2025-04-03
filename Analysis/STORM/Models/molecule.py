import uuid
import numpy as np

class Molecule:
    def __init__(
        self, molecule_id=uuid.uuid4().hex, experiment_id=0, start_track=None, end_track=None, 
        total_on_time=0, num_tracks=0, tracks=None
    ):
        """
        Initializes a Molecule object.

        Args:
            molecule_id (int): Unique molecule identifier.
            experiment_id (int): ID of the associated experiment.
            start_track (int, optional): Start track ID.
            end_track (int, optional): End track ID.
            total_on_time (float, optional): Total on-time of the molecule.
            num_tracks (int, optional): Number of tracks assigned to this molecule.
            tracks (list, optional): List of Track objects assigned to this molecule.
        """
        self.molecule_id = molecule_id
        self.experiment_id = experiment_id
        self.start_track = start_track
        self.end_track = end_track
        self.total_on_time = total_on_time
        self.num_tracks = num_tracks
        self.tracks = tracks if tracks is not None else []

    def add_track(self, track):
        """Add a track to the molecule and update properties."""
        self.tracks.append(track)
        self.num_tracks += 1
        self.total_on_time += track.on_time

        if self.start_track is None or track.track_id < self.start_track:
            self.start_track = track.track_id
        if self.end_track is None or track.track_id > self.end_track:
            self.end_track = track.track_id

    def compute_centroid(self):
        """Compute centroid of molecule based on tracks."""
        if not self.tracks:
            return None, None, None

        x_values = [track.compute_weighted_position()[0] for track in self.tracks]
        y_values = [track.compute_weighted_position()[1] for track in self.tracks]

        return np.mean(x_values), np.mean(y_values), 0

    def __repr__(self):
        return f"Molecule(ID={self.molecule_id}, Tracks={len(self.tracks)}, Total On Time={self.total_on_time})"
    
    def to_dict(self):
        return {
            "molecule_id": self.molecule_id,
            "experiment_id": self.experiment_id,
            "start_track": self.start_track,
            "end_track": self.end_track,
            "total_on_time": self.total_on_time,
            "num_tracks": self.num_tracks,
            "tracks": [track.to_dict() for track in self.tracks]
        }
