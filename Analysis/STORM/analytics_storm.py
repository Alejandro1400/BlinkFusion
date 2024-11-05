import pandas as pd

def obtain_molecules_metrics(molecules):

    # Calculate number of molecules
    num_molecules = molecules['MOLECULE_ID'].nunique()

    # Calculate mean switching cycles per molecule
    mean_switching_cycles = molecules['#_TRACKS'].mean()

    # Calculate mean on time per molecule
    mean_on_time = molecules['TOTAL_ON_TIME'].mean()

    # Calculate how many molecules are bleached
    bleached = molecules[molecules['BLEACHED'] == True]
    survival_fraction = (num_molecules - len(bleached)) / num_molecules

    # Calculate Duty Cycle
    duty_cycle = mean_on_time / 10000

    # Create metrics dataframe
    metrics = pd.DataFrame({
        'Molecules': [num_molecules],
        'Switching Cycles per Mol': [mean_switching_cycles],
        'On Time per SC': [mean_on_time],
        'Survival Fraction': [survival_fraction],
        'Duty Cycle': [duty_cycle]
    })

    return metrics


def calculate_duty_cycle(molecules, tracks, interval, total_frames):
    duty_cycles = {}
    time_bins = range(0, total_frames, interval)

    frame_rate = 30/1000  # 50 ms per frame

    start_frame_dict = dict(zip(tracks['TRACK_ID'], tracks['START_FRAME']))
    end_frame_dict = dict(zip(tracks['TRACK_ID'], tracks['END_FRAME']))

    # Map START_FRAME and END_FRAME for START_TRACK and END_TRACK onto the molecules DataFrame
    molecules['START_FRAME_start'] = molecules['START_TRACK'].map(start_frame_dict)
    molecules['END_FRAME_start'] = molecules['START_TRACK'].map(end_frame_dict)
    molecules['END_FRAME_end'] = molecules['END_TRACK'].map(end_frame_dict)

    for start_bin in time_bins:
        end_bin = start_bin + interval
        
        # Create dictionaries from tracks to map START_FRAME and END_FRAME for START_TRACK and END_TRACK
        # Filter molecules based on the given criteria
        filtered_molecules = molecules[
            (molecules['START_FRAME_start'] <= end_bin) &  # Molecule starts before or during the bin
            (
                (molecules['BLEACHED'] == False) |  # Molecule is not bleached
                ((molecules['BLEACHED'] == True) & (molecules['END_FRAME_end'] > end_bin))  # Molecule is bleached but ends after the bin
            )
        ]

        mol_id_list = filtered_molecules['MOLECULE_ID'].tolist()
        
        total_on_time = 0
        total_possible_time = len(filtered_molecules) * interval
        
        # For each molecule, find relevant tracks
        for mol_id in mol_id_list:
            # Find relevant tracks for this molecule
            relevant_tracks = tracks[(tracks['MOLECULE_ID'] == mol_id) &
                                    ((tracks['START_FRAME'] <= end_bin) & (tracks['START_FRAME'] >= start_bin) |
                                    (tracks['END_FRAME'] >= start_bin) & (tracks['END_FRAME'] <= end_bin))]
            
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

        # Convert to seconds the end_bin
        start_bin_seconds = start_bin * frame_rate

        duty_cycles[start_bin_seconds] = duty_cycle

    return pd.Series(duty_cycles)


def calculate_survival_fraction(molecules, tracks, interval, total_frames):
    survival_fractions = {}
    time_bins = range(0, total_frames, interval)

    frame_rate = 30/1000  # 50 ms per frame

    start_frame_dict = dict(zip(tracks['TRACK_ID'], tracks['START_FRAME']))
    end_frame_dict = dict(zip(tracks['TRACK_ID'], tracks['END_FRAME']))

    # Map START_FRAME and END_FRAME for START_TRACK and END_TRACK onto the molecules DataFrame
    molecules['START_FRAME_start'] = molecules['START_TRACK'].map(start_frame_dict)
    molecules['END_FRAME_start'] = molecules['START_TRACK'].map(end_frame_dict)
    molecules['END_FRAME_end'] = molecules['END_TRACK'].map(end_frame_dict)

    # Number of molecules 
    total_molecules = len(molecules)

    # Number of survivors as those where END_FRAME_end occurs before 60% of the total frames
    bleached_molecules = molecules[molecules['END_FRAME_end'] < 0.6 * total_frames]

    bleached_count = 0

    # Remove last bin from time_bins
    time_bins = time_bins[:-1]
    for start_bin in time_bins:
        end_bin = start_bin + interval  # Define the length of the bin
        
        # Filter molecules where END_FRAME_end occurs before 60% of the total frames
        bleached_interval = bleached_molecules[
            (bleached_molecules['START_FRAME_start'] > start_bin) &  # Molecule starts before or during the bin
            (bleached_molecules['END_FRAME_end'] <= end_bin)  # Molecule ends after the bin
        ]

        bleached_count += len(bleached_interval)
        
        # Calculate survival fraction for the bin
        survival_fraction = (total_molecules-bleached_count) / total_molecules if total_molecules > 0 else 0

        # Convert to seconds the end_bin
        end_bin_seconds = end_bin * frame_rate

        if start_bin == 0:
            survival_fractions[0] = 1

        survival_fractions[end_bin_seconds] = survival_fraction

    return pd.Series(survival_fractions)

