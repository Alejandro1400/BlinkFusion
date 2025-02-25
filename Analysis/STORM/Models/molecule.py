import numpy as np

class Molecule:
    def __init__(self, molecule_id):
        self.molecule_id = int(molecule_id)
        self.tracks = []
        self.start_track = None
        self.end_track = None
        self.total_on_time = 0
        self.num_tracks = 0

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
