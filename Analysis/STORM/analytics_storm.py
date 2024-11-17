import math
import numpy as np
import pandas as pd
import streamlit as st


import pandas as pd

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


def weighted_mean(data, weights):
    """Calculate weighted mean for a Series with given weights."""
    return (data * weights).sum() / weights.sum()

def aggregate_metrics(grouped_df):
    """Apply custom aggregation for each metric using weighted means where applicable."""
    aggregation = {
        'Population Mol': 'sum',
        'QE DC Population': 'sum',
        'QE Active Population': 'sum',
        # Replace 'mean' with a custom function that calculates the weighted mean:
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



def calculate_time_series_metrics(molecules, tracks, interval, total_frames, exposure_time):

    time_series_df = pd.DataFrame(columns=['Duty Cycle', 'Survival Fraction', 'Population Mol'])

    frame_rate = 1000/exposure_time
    interval_frames = int(interval*frame_rate)

    time_bins = range(0, total_frames, interval_frames)

    # Get the start and end frames for each track
    start_frame_dict = dict(zip(tracks['TRACK_ID'], tracks['START_FRAME']))
    end_frame_dict = dict(zip(tracks['TRACK_ID'], tracks['END_FRAME']))

    # Map START_FRAME and END_FRAME for START_TRACK and END_TRACK onto the molecules DataFrame
    molecules['START_FRAME_start'] = molecules['START_TRACK'].map(start_frame_dict)
    molecules['END_FRAME_start'] = molecules['START_TRACK'].map(end_frame_dict)
    molecules['END_FRAME_end'] = molecules['END_TRACK'].map(end_frame_dict) 

    total_molecules = len(molecules)

    for start_bin in time_bins:
        end_bin = start_bin + interval_frames

        # Filter molecules based on the given criteria
        past_molecules = molecules[molecules['START_FRAME_start'] <= end_bin]

        active_molecules = past_molecules[past_molecules['END_FRAME_end'] > end_bin]
        bleached_molecules = past_molecules[past_molecules['END_FRAME_end'] <= end_bin]

        
        survival_fraction = (total_molecules - len(bleached_molecules)) / total_molecules if total_molecules > 0 else 0
        population = 0


        mol_id_list = active_molecules['MOLECULE_ID'].tolist()
        
        total_on_time = 0
        total_possible_time = len(active_molecules) * interval_frames

        # For each molecule, find relevant tracks
        for mol_id in mol_id_list:
            # Find relevant tracks for this molecule
            relevant_tracks = tracks[(tracks['MOLECULE_ID'] == mol_id) &
                                    ((tracks['START_FRAME'] <= end_bin) & (tracks['START_FRAME'] >= start_bin) |
                                    (tracks['END_FRAME'] >= start_bin) & (tracks['END_FRAME'] <= end_bin))]
            
            if len(relevant_tracks) > 0:
                population += 1
            
            # Calculate on-time for each relevant track
            for _, track in relevant_tracks.iterrows():
                # Adjust start and end frame to the bin
                adjusted_start = max(track['START_FRAME'], start_bin)
                adjusted_end = min(track['END_FRAME'], end_bin)

                # Calculate the on-time for this track in the bin
                on_time = adjusted_end - adjusted_start
                total_on_time += on_time
        
        # Calculate duty cycle for the bin
        if total_possible_time > 0:
            duty_cycle = total_on_time / total_possible_time
        else:
            duty_cycle = 0

        end_bin_seconds = int(end_bin / frame_rate)

        time_series_df.loc[end_bin_seconds] = [duty_cycle, survival_fraction, population]

    return time_series_df


def calculate_molecule_metrics(selected_qe_tracks, qe_start, qe_end, exp):
    # Initialize dictionaries to store the results
    molecule_duty_cycle = {}
    molecule_photons = {}
    molecule_switching_cycles = {}
    track_intensity_within_range = []

    frame_rate = 1000/exp
    qe_start = int(qe_start*frame_rate)
    qe_end = int(qe_end*frame_rate)

    # Group by 'MOLECULE_ID' to process each molecule separately
    for molecule_id, group in selected_qe_tracks.groupby('MOLECULE_ID'):
        # Initialize metrics for the current molecule
        duty_cycle_sum = 0
        photons_sum = 0
        switching_cycles = len(group)

        for _, track in group.iterrows():
            start_frame = max(track['START_FRAME'], qe_start)  # Adjust START_FRAME to QE_START if necessary
            end_frame = min(track['END_FRAME'], qe_end)        # Adjust END_FRAME to QE_END if necessary

            # Calculate on-time within the quasi-equilibrium range
            on_time_within_range = max(0, end_frame - start_frame)
            print(on_time_within_range)
            duty_cycle_sum += on_time_within_range

            # Check if the track overlaps the quasi-equilibrium range for photons calculation
            if track['START_FRAME'] < qe_end and track['END_FRAME'] > qe_start:
                photons_sum += track['INTENSITY']

            # Check if the track is fully within the quasi-equilibrium range and add intensity
            if track['START_FRAME'] >= qe_start and track['END_FRAME'] <= qe_end:
                track_intensity_within_range.append(track['INTENSITY'])

        # Store calculated metrics for the molecule
        molecule_duty_cycle[molecule_id] = duty_cycle_sum/(qe_end - qe_start)
        molecule_photons[molecule_id] = photons_sum/switching_cycles
        molecule_switching_cycles[molecule_id] = switching_cycles

    # Convert results to DataFrames or Series for easy plotting
    duty_cycle_series = pd.Series(molecule_duty_cycle, name='Duty Cycle')
    photons_series = pd.Series(molecule_photons, name='Photons')
    switching_cycles_series = pd.Series(molecule_switching_cycles, name='Switching Cycles')
    intensity_series = pd.Series(track_intensity_within_range, name='Track Intensities')

    return duty_cycle_series, photons_series, switching_cycles_series, intensity_series
