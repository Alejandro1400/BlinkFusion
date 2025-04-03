import pandas as pd

class TimeSeriesCalculator:
    """Handles computations for time series analysis based on molecules and tracks."""

    def __init__(self, molecules, interval, total_frames, exposure_time):
        """
        Initialize the TimeSeriesCalculator.

        Args:
            molecules (list[Molecule]): List of Molecule objects containing Tracks.
            interval (int): Time interval in seconds for binning.
            total_frames (int): Total number of frames in the experiment.
            exposure_time (float): Exposure time in milliseconds.
        """
        self.molecules = molecules
        self.interval = interval
        self.total_frames = total_frames
        self.exposure_time = exposure_time
        self.frame_rate = 1000 / exposure_time
        self.interval_frames = int(interval * self.frame_rate)

    def calculate_time_series_metrics(self):
        """Calculates time series metrics and returns a DataFrame."""

        print(f"🔹 Starting Time Series Computation...\n")

        time_series_df = pd.DataFrame(columns=['Start Frame', 'End Frame', 'Duty Cycle', 'Survival Fraction', 'Population Mol', 
                                           'SC per Mol', 'On Time per SC (s)', 'Intensity per SC (Photons)'])

        total_molecules = len(self.molecules)

        time_bins = range(0, self.total_frames, self.interval_frames)

        for start_bin in time_bins:
            end_bin = start_bin + self.interval_frames

            # Get active molecules in this time bin
            past_molecules = [mol for mol in self.molecules if mol.tracks and min(t.start_frame for t in mol.tracks) <= end_bin]
            active_molecules = [mol for mol in past_molecules if any(t.end_frame > end_bin for t in mol.tracks)]
            bleached_molecules = [mol for mol in past_molecules if all(t.end_frame <= end_bin for t in mol.tracks)]

            survival_fraction = (total_molecules - len(bleached_molecules)) / total_molecules if total_molecules > 0 else 0
            population = len(active_molecules)

            # Compute metrics
            total_on_time = 0
            total_possible_time = population * self.interval_frames
            switching_cycles = 0
            total_intensity = 0
            total_tracks = 0

            for molecule in active_molecules:
                relevant_tracks = [t for t in molecule.tracks if
                                   (t.start_frame <= end_bin and t.start_frame >= start_bin) or
                                   (t.end_frame >= start_bin and t.end_frame <= end_bin)]
                
                if relevant_tracks:
                    switching_cycles += len(relevant_tracks)
                    total_tracks += len(relevant_tracks)

                for track in relevant_tracks:
                    adjusted_start = max(track.start_frame, start_bin)
                    adjusted_end = min(track.end_frame, end_bin)
                    on_time = adjusted_end - adjusted_start
                    total_on_time += on_time
                    total_intensity += max(track.intensity, 0)

            # Calculate metrics per molecule or per cycle
            duty_cycle = total_on_time / total_possible_time if total_possible_time > 0 else 0
            on_time_per_sc = total_on_time / total_tracks if total_tracks > 0 else 0
            sc_per_mol = switching_cycles / population if population > 0 else 0
            intensity_per_sc = total_intensity / total_tracks if total_tracks > 0 else 0

            end_bin_seconds = int(end_bin / self.frame_rate)

            time_series_df.loc[end_bin_seconds] = [start_bin, end_bin, duty_cycle, survival_fraction, population, sc_per_mol, on_time_per_sc, intensity_per_sc]

        return time_series_df
