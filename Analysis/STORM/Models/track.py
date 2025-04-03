import uuid
import numpy as np
from collections import defaultdict

import numpy as np

class Track:
    def __init__(
        self, track_id=uuid.uuid4().hex, experiment_id=0, start_frame=None, end_frame=None, intensity=0,
        offset=0, bkgstd=0, uncertainty=0, on_time=0, off_time=0, x=None, y=None,
        molecule_id=-1, gaps=None, gaps_counter=0, localizations=None
    ):
        """
        Initializes a Track object.

        Args:
            track_id (int): Unique track identifier.
            experiment_id (int): ID of the associated experiment.
            start_frame (int, optional): Start frame of the track.
            end_frame (int, optional): End frame of the track.
            intensity (float, optional): Intensity of the track.
            offset (float, optional): Offset value.
            bkgstd (float, optional): Background standard deviation.
            uncertainty (float, optional): Uncertainty value.
            on_time (float, optional): On-time duration.
            off_time (float, optional): Off-time duration.
            x (float, optional): Weighted centroid X coordinate.
            y (float, optional): Weighted centroid Y coordinate.
            molecule_id (int, optional): Assigned molecule ID (-1 by default).
            gaps (set, optional): Set of gap values.
            gaps_counter (int, optional): Number of gaps in the track.
            localizations (list, optional): List of localizations assigned to the track.
        """
        self.track_id = track_id
        self.experiment_id = experiment_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.intensity = intensity
        self.offset = offset
        self.bkgstd = bkgstd
        self.uncertainty = uncertainty
        self.on_time = on_time
        self.off_time = off_time
        self.x = x
        self.y = y
        self.molecule_id = molecule_id
        self.gaps = gaps if gaps is not None else set()
        self.gaps_counter = gaps_counter
        self.localizations = localizations if localizations is not None else []

    def add_localization(self, loc):
        """Add a localization to the track and update properties."""
        self.localizations.append(loc)

        # Update frame range
        if self.start_frame is None or loc.frame < self.start_frame:
            self.start_frame = loc.frame
        if self.end_frame is None or loc.frame > self.end_frame:
            self.end_frame = loc.frame

        # Update intensities and statistics
        self.intensity += loc.intensity
        self.offset += loc.offset
        self.bkgstd += loc.bkgstd
        self.uncertainty = np.mean([l.uncertainty for l in self.localizations])

        # Update weighted position
        self.update_weighted_position()

    def update_weighted_position(self):
        """Compute and update the weighted centroid of the track."""
        if not self.localizations:
            return None, None

        weights = np.array([loc.intensity for loc in self.localizations])
        x_coords = np.array([loc.x for loc in self.localizations])
        y_coords = np.array([loc.y for loc in self.localizations])

        self.x = np.sum(x_coords * weights) / np.sum(weights)
        self.y = np.sum(y_coords * weights) / np.sum(weights)

    def __repr__(self):
        return f"Track(ID={self.track_id}, Start={self.start_frame}, End={self.end_frame}, Localizations={len(self.localizations)})"
    
    def to_dict(self, embed_localizations=False):
        """
        Converts the Track object into a dictionary format suitable for MongoDB insertion.
        Optionally includes localizations based on `embed_localizations` flag.

        Args:
            embed_localizations (bool): If True, includes localizations in the output dictionary.
        """
        track_dict = {
            "id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "intensity": self.intensity,
            "offset": self.offset,
            "bkgstd": self.bkgstd,
            "uncertainty": self.uncertainty,
            "on_time": self.on_time,
            "off_time": self.off_time,
            "x": self.x,
            "y": self.y,
            "gaps": list(self.gaps),
            "molecule_id": self.molecule_id
        }
        if embed_localizations:
            track_dict['localizations'] = [loc.to_dict() for loc in self.localizations]
        return track_dict

