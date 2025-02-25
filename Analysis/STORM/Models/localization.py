import numpy as np

class Localization:
    def __init__(self, id_, frame, x, y, sigma, intensity, offset, bkgstd, uncertainty, track_id=0):
        self.id = int(id_)
        self.frame = int(frame)
        self.x = float(x)
        self.y = float(y)
        self.sigma = float(sigma)
        self.intensity = float(intensity)
        self.offset = float(offset)
        self.bkgstd = float(bkgstd)
        self.uncertainty = float(uncertainty)
        self.track_id = int(track_id)

    def distance_to(self, other):
        """Compute Euclidean distance to another localization."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"Localization(ID={self.id}, Frame={self.frame}, X={self.x}, Y={self.y}, TrackID={self.track_id})"
