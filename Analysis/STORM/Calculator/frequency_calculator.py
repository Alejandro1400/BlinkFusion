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