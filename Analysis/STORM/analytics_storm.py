import pandas as pd
import streamlit as st

def obtain_molecules_metrics(molecules, tracks, time_series, exposure_time):

    duty_cycles = time_series['Duty Cycle']
    duty_cycles = pd.Series(duty_cycles)

    frame_rate = 1000/exposure_time

    quasi_equilibrium = calculate_quasi_equilibrium(duty_cycles)
    quasi_equilibrium_data = time_series.loc[quasi_equilibrium]

    quasi_equilibrium_tracks = tracks[(tracks['START_FRAME'] <= int(quasi_equilibrium[-1]*frame_rate)) & (tracks['END_FRAME'] >= int(quasi_equilibrium[0]*frame_rate))]
    print(quasi_equilibrium_tracks)

    grouped = quasi_equilibrium_tracks.groupby('MOLECULE_ID')['INTENSITY']
    intensity_sums = grouped.sum()
    switching_cycles = grouped.count()
    mean_intensity = intensity_sums/switching_cycles
    # Calculate mean of these intensity sums
    qe_photons = mean_intensity.mean() if not mean_intensity.empty else 0

    qe_dc_population = (quasi_equilibrium_data['Population Mol']).sum()
    qe_duty_cycle = (quasi_equilibrium_data['Duty Cycle'] * quasi_equilibrium_data['Population Mol']).sum() / qe_dc_population
    print(f"Duty Cycle: {qe_duty_cycle}")
    print(quasi_equilibrium_data['Survival Fraction'])
    qe_survival_fraction = quasi_equilibrium_data['Survival Fraction'].iloc[-1]
    print(f"Survival Fraction: {qe_survival_fraction}")
    qe_act_population = quasi_equilibrium_tracks['MOLECULE_ID'].nunique()
    print(f"Active Population: {qe_act_population}")
    qe_switching_cycles = len(quasi_equilibrium_tracks)/qe_act_population
    print(f"Switching Cycles: {qe_switching_cycles}")
    #qe_photons = quasi_equilibrium_tracks['INTENSITY'].sum()/len(quasi_equilibrium_tracks)
    print(f"Photons: {qe_photons}")
    qe_uncertainty = quasi_equilibrium_tracks['UNCERTAINTY'].sum()/len(quasi_equilibrium_tracks)
    print(f"Uncertainty: {qe_uncertainty}")
    qe_on_time = quasi_equilibrium_tracks['ON_TIME'].sum()/len(quasi_equilibrium_tracks)
    print(f"On Time: {qe_on_time}")

    # Calculate number of molecules
    num_molecules = molecules['MOLECULE_ID'].nunique()

    # Calculate mean switching cycles per molecule
    mean_switching_cycles = molecules['#_TRACKS'].mean()

    # Calculate mean on time per molecule
    mean_on_time = molecules['TOTAL_ON_TIME'].mean()

    # Create metrics dataframe
    metrics = pd.DataFrame({
        'Molecules': [num_molecules],
        'Switching Cycles per mol': [mean_switching_cycles],
        'On Time per SC': [mean_on_time],
        'QE Duty Cycle': [qe_duty_cycle],
        'QE Survival Fraction': [qe_survival_fraction],
        'QE DC Population': [qe_dc_population],
        'QE Active Population': [qe_act_population],
        'QE Switching Cycles per mol': [qe_switching_cycles],
        'QE Photons per SC': [qe_photons],
        'QE Mean Uncertainty': [qe_uncertainty],
        'QE On Time per SC': [qe_on_time],
        'QE Start': [quasi_equilibrium[0]],
        'QE End': [quasi_equilibrium[-1]]
    })

    print(metrics)

    return metrics, quasi_equilibrium_tracks


def calculate_quasi_equilibrium(duty_cycles):
    # Ensure duty_cycles is a pandas Series
    if not isinstance(duty_cycles, pd.Series):
        duty_cycles = pd.Series(duty_cycles)

    # Initialize the best score to a high value
    min_score = float('inf')
    min_index = 0

    # Iterate through the series, checking segments of four consecutive values
    for i in range(len(duty_cycles) - 3):
        current_segment = duty_cycles[i:i+4]

        # Check to ensure no duty cycle value is zero in the segment
        if all(current_segment > 0):
            # Calculate the standard deviation of the segment
            segment_std = current_segment.std()

            # Update the best (minimum) score found so far
            if segment_std < min_score:
                min_score = segment_std
                min_index = i

    # Return the time indices of the 4 values that represent the quasi-equilibrium
    quasi_equilibrium_values = duty_cycles.index[min_index:min_index+4].tolist()

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
