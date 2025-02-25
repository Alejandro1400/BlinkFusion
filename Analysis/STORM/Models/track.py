import numpy as np
from collections import defaultdict

class Track:
    def __init__(self, track_id):
        self.track_id = int(track_id)
        self.localizations = []
        self.start_frame = None
        self.end_frame = None
        self.intensity = 0
        self.offset = 0
        self.bkgstd = 0
        self.uncertainty = 0
        self.gaps = set()
        self.molecule_id = -1  # Default before assignment
        self.on_time = 0
        self.off_time = 0

    def add_localization(self, loc):
        """Add a localization to the track and update properties."""
        self.localizations.append(loc)
        if self.start_frame is None or loc.frame < self.start_frame:
            self.start_frame = loc.frame
        if self.end_frame is None or loc.frame > self.end_frame:
            self.end_frame = loc.frame

        self.intensity += loc.intensity
        self.offset += loc.offset
        self.bkgstd += loc.bkgstd
        self.uncertainty = np.mean([l.uncertainty for l in self.localizations])

    def compute_weighted_position(self):
        """Compute weighted centroid of the track."""
        if not self.localizations:
            return None, None, None

        weights = np.array([loc.intensity for loc in self.localizations])
        x_coords = np.array([loc.x for loc in self.localizations])
        y_coords = np.array([loc.y for loc in self.localizations])

        x_weighted = np.sum(x_coords * weights) / np.sum(weights)
        y_weighted = np.sum(y_coords * weights) / np.sum(weights)

        return x_weighted, y_weighted, 0  # Z is 0 for now

    def __repr__(self):
        return f"Track(ID={self.track_id}, Start={self.start_frame}, End={self.end_frame}, Localizations={len(self.localizations)})"
