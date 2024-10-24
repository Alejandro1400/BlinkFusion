import pandas as pd
import numpy as np

def prepare_data(df):

    # Convert columns to uppercase
    df.columns = df.columns.str.upper()
    # Only keep the columns we need: Id, Track, X, Y, Frame, Quality, Total Intensity, SNR, Molecule_ID, Date, Sample and Image
    df = df[['DATE', 'SAMPLE', 'IMAGE', 'MOLECULE_ID', 'TRACK_ID', 'ID', 'X', 'Y', 'Z', 'FRAME', 'QUALITY', 'TOTAL_INTENSITY_CH1', 'SNR_CH1']]

    # Drop rows where TRACK_ID is NaN
    df = df.dropna(subset=['TRACK_ID'])

    return df


import pandas as pd

def bleaching_identification(df):
    # Ensure DataFrame is sorted properly for correct sequence operations
    df = df.sort_values(by=['DATE', 'SAMPLE', 'IMAGE', 'MOLECULE_ID', 'TRACK_ID', 'FRAME'])

    # Get the last track id for each molecule
    df['last_track_id'] = df.groupby(['DATE', 'SAMPLE', 'IMAGE', 'MOLECULE_ID'])['TRACK_ID'].transform('last')

    # Identify the end of blinks
    df['blink_end'] = df['TRACK_ID'] == df['last_track_id']

    print(df.head())

    # Group by DATE, SAMPLE, IMAGE, ID, MOLECULE_ID, TRACK_ID and calculate the first and last frame for each track
    track_frames = df.groupby(['DATE', 'SAMPLE', 'IMAGE', 'MOLECULE_ID', 'TRACK_ID']).agg({
        'FRAME': ['first', 'last'], 'blink_end': 'first'
    }).reset_index()

    print(track_frames.head())
    
    # Rename columns for clarity
    track_frames.columns = ['DATE', 'SAMPLE', 'IMAGE', 'MOLECULE_ID', 'TRACK_ID', 'FIRST_FRAME', 'LAST_FRAME', 'blink_end']

    # Calculate the maximum frame for each image
    max_frame_per_image = df.groupby(['DATE', 'SAMPLE', 'IMAGE'])['FRAME'].max().reset_index()
    max_frame_per_image.rename(columns={'FRAME': 'MAX_FRAME'}, inplace=True)

    # Merge max frames back into the track frames
    track_frames = pd.merge(track_frames, max_frame_per_image, on=['DATE', 'SAMPLE', 'IMAGE'])

    # Calculate off times within the tracks
    track_frames['NEXT_FIRST_FRAME'] = track_frames.groupby(['DATE', 'SAMPLE', 'IMAGE', 'MOLECULE_ID'])['FIRST_FRAME'].shift(-1)
    track_frames['OFF_TIME'] = (track_frames['NEXT_FIRST_FRAME'] - track_frames['LAST_FRAME']).where(~track_frames['blink_end'])

    # Calculate mean off time per image, considering only non-'blink_end' tracks
    mean_off_time_per_image = track_frames.groupby(['DATE', 'SAMPLE', 'IMAGE'])['OFF_TIME'].mean().reset_index()
    mean_off_time_per_image.rename(columns={'OFF_TIME': 'MEAN_OFF_TIME'}, inplace=True)

    # Merge mean off times back into the track frames
    track_frames = pd.merge(track_frames, mean_off_time_per_image, on=['DATE', 'SAMPLE', 'IMAGE'])

    # Determine if bleached
    track_frames['bleached'] = (track_frames['blink_end']) & ((track_frames['MAX_FRAME'] - track_frames['LAST_FRAME'] > track_frames['MEAN_OFF_TIME']))

    print(track_frames)

    return track_frames


def trackmate_blink_statistics(df):

    df = prepare_data(df)

    df = bleaching_identification(df)

    return df