import pandas as pd

from Analysis.STORM.Models.molecule import Molecule


def calculate_frequency(selected_qe_molecules, frames, qe_start, qe_end, exp, population='quasi', metric='molecule'):
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

    # Filter tracks based on QE limits
    filtered_molecule_list = filter_tracks_by_qe_period(
        molecule_list=selected_qe_molecules,
        qe_start=qe_start,
        qe_end=qe_end,
        population=population
    )

    for mol in filtered_molecule_list:
        on_time_sum = 0
        photons_sum = 0
        photons_track_count = 0
        switching_cycles = len(mol.tracks)

        for track in mol.tracks:
            start_frame = track.start_frame
            end_frame = track.end_frame

            if start_frame is None or end_frame is None:
                continue

            if population == 'quasi':
                start_frame = max(start_frame, qe_start)
                end_frame = min(end_frame, qe_end)

            on_time_frames = max(0, end_frame - start_frame)
            on_time_sec = on_time_frames / frame_rate

            on_time_sum += on_time_sec

            if on_time_frames != 0:
                if metric == 'molecule':
                    photons_sum += track.intensity
                    photons_track_count += 1
                elif metric == 'track':
                    results['photons'].append(track.intensity)
                    results['on_time'].append(on_time_sec)

        # Calculate metrics
        if population == 'quasi':
            duty_cycle = on_time_sum / ((qe_end - qe_start) / frame_rate) if (qe_end - qe_start) > 0 else 0
        else:
            duty_cycle = on_time_sum / (frames * frame_rate) if frames > 0 else 0

        if metric == 'molecule':
            photons = photons_sum / photons_track_count if photons_track_count > 0 else 0
            on_time = on_time_sum

            results['photons'].append(photons)
            results['on_time'].append(on_time)

        results['duty_cycle'].append(duty_cycle)
        results['switching_cycles'].append(switching_cycles)

        # Classification logic
        classification = ""
        if switching_cycles == 1:
            classification = "Blinks On Once" if duty_cycle <= 0.5 else "Blinks Off Once"
        elif switching_cycles == 2:
            classification = "Blinks On Mult. Times" if duty_cycle <= 0.5 else "Blinks Off Once"
        elif switching_cycles > 2:
            classification = "Blinks On Mult. Times" if duty_cycle <= 0.25 else "Blinks Off Mult. Times"
        else:
            classification = "Uncharacterized"

        # Bleaching detection
        last_track = mol.tracks[-1] if mol.tracks else None
        bleaching = "NA"
        if last_track:
            if last_track.track_id == mol.end_track:
                if last_track.end_frame == frames:
                    bleaching = "Inverse Photobleach"
                else:
                    bleaching = "Photobleach"

        classification_data[classification].append({
            "Tracks": [t.track_id for t in mol.tracks],
            "Bleaching": bleaching,
            "Duty Cycle": duty_cycle,
        })

    # Convert results to DataFrames or Series for easy plotting and analysis
    duty_cycle_series = pd.Series(results['duty_cycle'], name='Duty Cycle')
    photons_series = pd.Series(results['photons'], name='Intensity per SC (Photons)')
    switching_cycles_series = pd.Series(results['switching_cycles'], name='Switching Cycles')
    on_time_series = pd.Series(results['on_time'], name='On Time per SC (s)')

    return duty_cycle_series, photons_series, switching_cycles_series, on_time_series, classification_data


def filter_tracks_by_qe_period(molecule_list, qe_start, qe_end, population='whole'):
    """
    Filters tracks inside molecules based on Quasi-Equilibrium (QE) period.

    Args:
        molecule_list (list of Molecule): The list of Molecule objects to process.
        qe_start (int): Start of QE period (in frames).
        qe_end (int): End of QE period (in frames).
        population (str): 'quasi' or 'whole'. If 'whole', all tracks are kept.

    Returns:
        list of Molecule: New list of Molecule objects with filtered tracks.
    """
    filtered_molecules = []

    for mol in molecule_list:
        if population == 'quasi':
            filtered_tracks = [
                track for track in mol.tracks
                if track.start_frame is not None and track.end_frame is not None
                and track.start_frame <= qe_end and track.end_frame >= qe_start
            ]
        else:
            filtered_tracks = mol.tracks  # keep all tracks

        if filtered_tracks:
            # Optionally create a new molecule (to avoid modifying original)
            new_mol = Molecule(
                molecule_id=mol.molecule_id,
                experiment_id=mol.experiment_id,
                start_track=filtered_tracks[0].track_id,
                end_track=filtered_tracks[-1].track_id,
                total_on_time=sum(t.on_time for t in filtered_tracks),
                num_tracks=len(filtered_tracks),
                tracks=filtered_tracks
            )
            filtered_molecules.append(new_mol)

    return filtered_molecules