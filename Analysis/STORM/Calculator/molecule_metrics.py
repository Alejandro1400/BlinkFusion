import math
import numpy as np
import pandas as pd
import streamlit as st

class MoleculeMetrics:
    """
    A class to compute molecular metrics and aggregate statistics for experimental data.
    """

    def __init__(self, grouped_molecules, time_series_dict, metadata):
        """
        Initialize the class with required datasets.

        Args:
            grouped_molecules (dict): {experiment_id: {molecule_id: Molecule object}}
            time_series_dict (dict): {experiment_id: time-series DataFrame}
            metadata (dict): {experiment_id: metadata dictionary}
        """
        self.grouped_molecules = grouped_molecules
        self.time_series_dict = time_series_dict
        self.metadata = metadata

        # Precompute frame rates for each experiment
        self.frame_rates = {
            exp_id: 1000 / self.metadata[exp_id].get("Exposure", 1)
            for exp_id in self.metadata
        }
    

    def calculate_quasi_equilibrium(self, duty_cycles):
        """
        Calculate the quasi-equilibrium period by finding the segment with the lowest variation.

        Args:
            duty_cycles (pd.Series): A pandas Series containing duty cycle values indexed by time.

        Returns:
            list: A list of time indices representing the quasi-equilibrium period.
        """
        if not isinstance(duty_cycles, pd.Series):
            duty_cycles = pd.Series(duty_cycles)

        min_score = float('inf')
        min_index = 0

        for i in range(len(duty_cycles) - 4):
            current_segment = duty_cycles[i:i+5]

            if all(current_segment > 0):
                segment_std = current_segment.std()
                if segment_std < min_score:
                    min_score = segment_std
                    min_index = i

        return duty_cycles.index[min_index:min_index+5].tolist()


    def obtain_molecules_metrics(self):
        """
        Compute molecular metrics based on molecule objects and time-series data.

        Returns:
            pd.DataFrame: A dataframe containing calculated molecular metrics.
        """
        results = []

        for experiment_id, molecules in self.grouped_molecules.items():
            if experiment_id not in self.time_series_dict:
                continue  # Skip if no time-series data

            time_series_df = self.time_series_dict[experiment_id]
            duty_cycles = pd.Series(time_series_df['Duty Cycle'])

            # Get frame rate from metadata
            frame_rate = self.frame_rates.get(experiment_id, 1)

            population = len(molecules)
            intensity_mol = sum(
                sum(track.intensity for track in mol.tracks) for mol in molecules
            ) / population if population > 0 else 0

            switching_cycles_mol = sum(
                len(mol.tracks) for mol in molecules
            ) / population if population > 0 else 0
            
            intensity_sc = sum(
                sum(track.intensity for track in mol.tracks) for mol in molecules
            ) / sum(len(mol.tracks) for mol in molecules) if population > 0 else 0

            uncertainty = sum(
                sum(track.uncertainty for track in mol.tracks) for mol in molecules
            ) / sum(len(mol.tracks) for mol in molecules) if population > 0 else 0

            on_time = sum(
                sum(track.on_time for track in mol.tracks) for mol in molecules
            ) / sum(len(mol.tracks) for mol in molecules) / frame_rate if population > 0 else 0

            # Quasi-Equilibrium Period Calculation
            quasi_equilibrium = self.calculate_quasi_equilibrium(duty_cycles)
            quasi_equilibrium_data = time_series_df.loc[quasi_equilibrium]
            quasi_equilibrium_data['Start Time (s)'] = quasi_equilibrium_data['Start Frame'] / frame_rate
            quasi_equilibrium_data['End Time (s)'] = quasi_equilibrium_data['End Frame'] / frame_rate

            # Get the earliest start time and latest end time
            earliest_start_time = quasi_equilibrium_data['Start Frame'].iloc[0] if not quasi_equilibrium_data.empty else None
            latest_end_time = quasi_equilibrium_data['End Frame'].iloc[-1] if not quasi_equilibrium_data.empty else None

            # Filter molecules based on quasi-equilibrium time
            quasi_equilibrium_molecules = [
                mol for mol in molecules if (any(
                    track.end_frame >= earliest_start_time and track.track_id == mol.end_track for track in mol.tracks
                ) and any(
                    track.start_frame <= latest_end_time and track.track_id == mol.start_track for track in mol.tracks
                ))
            ]

            for mol in quasi_equilibrium_molecules:
                mol.tracks = [
                    track for track in mol.tracks
                    if track.start_frame <= latest_end_time and track.end_frame >= earliest_start_time
                ]

            # Aggregate intensities for quasi-equilibrium molecules
            qe_photons_mol = sum(
                sum(track.intensity for track in mol.tracks) for mol in quasi_equilibrium_molecules
            ) / len(quasi_equilibrium_molecules) if quasi_equilibrium_molecules else 0

            qe_photons_track = (
                sum(track.intensity for mol in quasi_equilibrium_molecules for track in mol.tracks) /
                sum(len(mol.tracks) for mol in quasi_equilibrium_molecules)
            ) if quasi_equilibrium_molecules and any(mol.tracks for mol in quasi_equilibrium_molecules) else 0
            

            qe_switching_cycles = sum(
                len(mol.tracks) for mol in quasi_equilibrium_molecules
            ) / len(quasi_equilibrium_molecules) if quasi_equilibrium_molecules else 0
            
            qe_uncertainty = sum(
                sum(track.uncertainty for track in mol.tracks) for mol in quasi_equilibrium_molecules
            ) / sum(len(mol.tracks) for mol in quasi_equilibrium_molecules) if quasi_equilibrium_molecules else 0
            
            qe_on_time = sum(
                sum(track.on_time for track in mol.tracks) for mol in quasi_equilibrium_molecules
            ) / sum(len(mol.tracks) for mol in quasi_equilibrium_molecules) / frame_rate if quasi_equilibrium_molecules else 0

            qe_dc_population = quasi_equilibrium_data['Population Mol'].sum()
            qe_duty_cycle = (quasi_equilibrium_data['Duty Cycle'] * quasi_equilibrium_data['Population Mol']).sum() / qe_dc_population
            qe_survival_fraction = quasi_equilibrium_data['Survival Fraction'].iloc[-1] if not quasi_equilibrium_data.empty else 0

            midpoint_index = len(time_series_df) // 2
            midway_survival_fraction = time_series_df['Survival Fraction'].iloc[midpoint_index] if len(time_series_df) > 0 else 0

            metrics_dict = {
                'Experiment ID': experiment_id,
                'Population Mol': population,
                'QE Active Population': len(quasi_equilibrium_molecules),
                'QE Duty Cycle': qe_duty_cycle,
                'QE Survival Fraction': qe_survival_fraction,
                'Mid Survival Fraction': midway_survival_fraction,
                'Int. per Mol (Photons)': intensity_mol,
                'Int. per SC (Photons)': intensity_sc,
                'QE Int. per Mol (Photons)': qe_photons_mol,
                'QE Int. per SC (Photons)': qe_photons_track,
                'SC per Mol': switching_cycles_mol,
                'QE SC per Mol': qe_switching_cycles,
                'On Time per SC (s)': on_time,
                'QE On Time per SC (s)': qe_on_time,
                'Uncertainty (um)': uncertainty,
                'QE Uncertainty (um)': qe_uncertainty,
                'QE Period (s)': f"{math.ceil(quasi_equilibrium_data['Start Time (s)'].iloc[0] / 10) * 10}-{math.ceil(quasi_equilibrium_data['Start Time (s)'].iloc[-1] / 10) * 10}"
            }
            results.append(metrics_dict)

        return pd.DataFrame(results)

    def aggregate_metrics(self, grouped_df):
        """
        Apply custom aggregation for each metric using weighted means where applicable.

        Args:
            grouped_df (pd.DataFrame): The dataframe grouped by identifier.

        Returns:
            pd.DataFrame: Aggregated results.
        """
        aggregation = {
            'Population Mol': 'sum',
            'QE Active Population': 'sum',
            'QE Duty Cycle': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'QE Survival Fraction': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'Mid Survival Fraction': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'Int. per Mol (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'Int. per SC (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE Int. per Mol (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'QE Int. per SC (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'SC per Mol': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE SC per Mol': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'On Time per SC (s)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE On Time per SC (s)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'Uncertainty (um)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE Uncertainty (um)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'QE Period (s)': lambda x: list(x.unique()),
            '# Images': 'count'
        }
        return grouped_df.agg(aggregation)
