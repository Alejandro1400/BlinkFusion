import math
import numpy as np
import pandas as pd
import streamlit as st

class MoleculeMetricsCalculator:
    """Handles computations related to molecule-level metrics."""

    def obtain_molecules_metrics(tracks, time_series, metadata):
        results = []
        
        for identifier in metadata['IDENTIFIER'].unique():
            tracks_df = tracks[tracks['IDENTIFIER'] == identifier]
            time_series_df = time_series[time_series['IDENTIFIER'] == identifier]

            if time_series_df.empty:
                continue

            duty_cycles = time_series_df['Duty Cycle']
            duty_cycles = pd.Series(duty_cycles)

            exposure_time = metadata[metadata['IDENTIFIER'] == identifier]['EXPOSURE'].iloc[0]
            frame_rate = 1000 / exposure_time

            population = len(tracks_df['MOLECULE_ID'].unique()) if not tracks_df.empty else 0
            intensity_mol = tracks_df['INTENSITY'].sum() / population if population > 0 else 0
            switching_cycles_mol = len(tracks_df) / population if population > 0 else 0
            intensity_sc = tracks_df['INTENSITY'].sum() / len(tracks_df) if len(tracks_df) > 0 else 0
            uncertainty = tracks_df['UNCERTAINTY'].sum() / len(tracks_df) if len(tracks_df) > 0 else 0
            on_time = (tracks_df['ON_TIME'].sum() / len(tracks_df)) / frame_rate if len(tracks_df) > 0 else 0

            quasi_equilibrium = calculate_quasi_equilibrium(duty_cycles)
            quasi_equilibrium_data = time_series_df.loc[quasi_equilibrium]

            quasi_equilibrium_tracks = tracks_df[
                (tracks_df['START_FRAME'] <= int(quasi_equilibrium[-1] * frame_rate)) & 
                (tracks_df['END_FRAME'] >= int(quasi_equilibrium[0] * frame_rate))
            ]

            grouped = quasi_equilibrium_tracks.groupby('MOLECULE_ID')['INTENSITY']
            intensity_sums = grouped.sum()
            switching_cycles = grouped.count()

            qe_photons_mol = intensity_sums.mean() if not intensity_sums.empty else 0
            mean_intensity = intensity_sums / switching_cycles
            qe_photons = mean_intensity.mean() if not mean_intensity.empty else 0

            qe_dc_population = (quasi_equilibrium_data['Population Mol']).sum()
            qe_duty_cycle = (quasi_equilibrium_data['Duty Cycle'] * quasi_equilibrium_data['Population Mol']).sum() / qe_dc_population
            qe_survival_fraction = quasi_equilibrium_data['Survival Fraction'].iloc[-1] if not quasi_equilibrium_data.empty else 0
            qe_act_population = quasi_equilibrium_tracks['MOLECULE_ID'].nunique()
            qe_switching_cycles = len(quasi_equilibrium_tracks) / qe_act_population if qe_act_population > 0 else 0
            qe_uncertainty = quasi_equilibrium_tracks['UNCERTAINTY'].sum() / len(quasi_equilibrium_tracks) if len(quasi_equilibrium_tracks) > 0 else 0
            qe_on_time = (quasi_equilibrium_tracks['ON_TIME'].sum() / len(quasi_equilibrium_tracks))/frame_rate if len(quasi_equilibrium_tracks) > 0 else 0

            # Calculate Midway Survival Fraction
            midpoint_index = len(time_series_df) // 2
            midway_survival_fraction = time_series_df['Survival Fraction'].iloc[midpoint_index] if len(time_series_df) > 0 else 0

            metrics_dict = {
                'IDENTIFIER': identifier,
                'Population Mol': population,
                'QE DC Population': qe_dc_population,
                'QE Active Population': qe_act_population,
                'QE Duty Cycle': qe_duty_cycle,
                'QE Survival Fraction': qe_survival_fraction,
                'Mid Survival Fraction': midway_survival_fraction,
                'Int. per Mol (Photons)': intensity_mol,
                'Int. per SC (Photons)': intensity_sc,
                'QE Int. per Mol (Photons)': qe_photons_mol,
                'QE Int. per SC (Photons)': qe_photons,
                'SC per Mol': switching_cycles_mol,
                'QE SC per Mol': qe_switching_cycles,
                'On Time per SC (s)': on_time,
                'QE On Time per SC (s)': qe_on_time,
                'Uncertainty (um)': uncertainty,
                'QE Uncertainty (um)': qe_uncertainty,
                'QE Period (s)': f"{math.ceil(quasi_equilibrium[0] / 10) * 10}-{math.ceil(quasi_equilibrium[-1] / 10) * 10}"
            }
            results.append(metrics_dict)

        # Convert results list to DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def aggregate_metrics(grouped_df):
        """Apply custom aggregation for each metric using weighted means where applicable."""
        aggregation = {
            'Population Mol': 'sum',
            'QE DC Population': 'sum',
            'QE Active Population': 'sum',
            'QE Duty Cycle': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE DC Population']),
            'QE Survival Fraction': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE DC Population']),
            'Mid Survival Fraction': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'Int. per Mol (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'Int. per SC (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE Int. per Mol (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE DC Population']),
            'QE Int. per SC (Photons)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'SC per Mol': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE SC per Mol': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'On Time per SC (s)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE On Time per SC (s)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'Uncertainty (um)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'Population Mol']),
            'QE Uncertainty (um)': lambda x: np.average(x, weights=grouped_df.loc[x.index, 'QE Active Population']),
            'QE Period (s)': lambda x: list(x.unique()), # Collect unique periods
            '# Images': 'count'
        
        }
        return grouped_df.agg(aggregation)




    def calculate_quasi_equilibrium(duty_cycles):
        # Ensure duty_cycles is a pandas Series
        if not isinstance(duty_cycles, pd.Series):
            duty_cycles = pd.Series(duty_cycles)

        # Initialize the best score to a high value
        min_score = float('inf')
        min_index = 0

        # Iterate through the series, checking segments of four consecutive values
        for i in range(len(duty_cycles) - 4):
            current_segment = duty_cycles[i:i+5]

            # Check to ensure no duty cycle value is zero in the segment
            if all(current_segment > 0):
                # Calculate the standard deviation of the segment
                segment_std = current_segment.std()

                # Update the best (minimum) score found so far
                if segment_std < min_score:
                    min_score = segment_std
                    min_index = i

        quasi_equilibrium_values = duty_cycles.index[min_index:min_index+5].tolist()

        return quasi_equilibrium_values



def calculate_frequency(selected_qe_tracks, selected_qe_molecules, frames, qe_start, qe_end, exp, population='quasi', metric='molecule'):
    """
    Calculates frequency-related metrics for tracks and molecules, including duty cycle, photons, 
    switching cycles, on-time, and classifies molecules into predefined categories.

    Args:
        selected_qe_tracks (pd.DataFrame): DataFrame containing track information.
        selected_qe_molecules (pd.DataFrame): DataFrame containing molecule information.
        selected_qe_localizations (pd.DataFrame): DataFrame containing localization data.
        frames (int): Total number of frames in the acquisition.
        qe_start (float): Start time for quasi-equilibrium in seconds.
        qe_end (float): End time for quasi-equilibrium in seconds.
        exp (float): Exposure time in milliseconds.
        population (str): 'quasi' for quasi-equilibrium analysis, 'whole' for entire population.
        metric (str): 'molecule' or 'track' to define grouping level for analysis.

    Returns:
        tuple: Duty cycle, photons, switching cycles, on-time, classifications, and classified IDs and tracks.
    """
    # Initialize dictionaries to store results
    results = {
        'duty_cycle': [],
        'photons': [],
        'switching_cycles': [],
        'on_time': []
    }
    classification_data = {
        "Blinks On Once": [],
        "Blinks Off Once": [],
        "Blinks On Mult. Times": [],
        "Blinks Off Mult. Times": [],
        "Uncharacterized": []
    }

    frame_rate = 1000 / exp
    qe_start = int(qe_start * frame_rate)
    qe_end = int(qe_end * frame_rate)

    if population == 'quasi':
        selected_qe_tracks = selected_qe_tracks[
            (selected_qe_tracks['START_FRAME'] <= qe_end) & (selected_qe_tracks['END_FRAME'] >= qe_start)
        ]

    for molecule_id, molecule_group in selected_qe_tracks.groupby('MOLECULE_ID'):
        on_time_sum = 0
        photons_sum = 0
        photons_track_count = 0
        switching_cycles = len(molecule_group)

        for _, track in molecule_group.iterrows():
            start_frame = track['START_FRAME']
            end_frame = track['END_FRAME']
            if population == 'quasi':
                start_frame = max(start_frame, qe_start)
                end_frame = min(end_frame, qe_end)

            # Calculate on-time within the range
            on_time_within_range = max(0, end_frame - start_frame)
            on_time_sum += on_time_within_range

            if on_time_within_range != 0:
                if metric == 'molecule':
                    photons_sum += track['INTENSITY']
                    photons_track_count += 1
                elif metric == 'track':
                    results['photons'].append(track['INTENSITY'])
                    results['on_time'].append(on_time_within_range / frame_rate)

        # Calculate metrics
        if population == 'quasi':
            duty_cycle = on_time_sum / (qe_end - qe_start) if (qe_end - qe_start) > 0 else 0
        else:
            duty_cycle = on_time_sum / frames if frames > 0 else 0

        if metric == 'molecule':
            photons = photons_sum / photons_track_count if photons_track_count > 0 else 0
            on_time = on_time_sum / len(molecule_group) if len(molecule_group) > 0 else 0

            results['photons'].append(photons)
            results['on_time'].append(on_time)

        results['duty_cycle'].append(duty_cycle)
        results['switching_cycles'].append(switching_cycles)

        # Classification logic
        classification = ""
        if len(molecule_group) == 1:  # Single track
            track = molecule_group.iloc[0]
            if duty_cycle <= 0.5:
                classification = "Blinks On Once"
            else:
                classification = "Blinks Off Once"
        elif len(molecule_group) == 2:  # Multiple tracks
            if duty_cycle <= 0.5:
                classification = "Blinks On Mult. Times"
            else:
                classification = "Blinks Off Once"
        elif len(molecule_group) > 2:  # Multiple tracks
            if duty_cycle <= 0.25:
                classification = "Blinks On Mult. Times"
            else:
                classification = "Blinks Off Mult. Times"
        else:
            classification = "Uncharacterized"

        # Check photobleaching status
        last_track = molecule_group.iloc[-1]
        if last_track['TRACK_ID'] == selected_qe_molecules.loc[selected_qe_molecules['MOLECULE_ID'] == molecule_id, 'END_TRACK'].iloc[0]:
            if last_track['END_FRAME'] == frames:  # Ends in last frame
                bleaching = "Inverse Photobleach" 
            else:  # Ends before last frame
                bleaching = "Photobleach"
        else:
            bleaching = "NA"

        classification_data[classification].append({
            "Tracks": molecule_group['TRACK_ID'].tolist(),
            "Bleaching": bleaching,
            "Duty Cycle": duty_cycle,
        })

    # Convert results to DataFrames or Series for easy plotting and analysis
    duty_cycle_series = pd.Series(results['duty_cycle'], name='Duty Cycle')
    photons_series = pd.Series(results['photons'], name='Intensity per SC (Photons)')
    switching_cycles_series = pd.Series(results['switching_cycles'], name='Switching Cycles')
    on_time_series = pd.Series(results['on_time'], name='On Time per SC (s)')

    return duty_cycle_series, photons_series, switching_cycles_series, on_time_series, classification_data
