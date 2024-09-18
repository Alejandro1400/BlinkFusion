import pandas as pd

def calculate_summarized_metrics(data, type_to_analyze, columns_to_exclude):

    # Define the aggregation methods for each column
    aggregations = {
        'Network': lambda x: x[x != 0].nunique(),
        'Contour': 'count',
        'Length': 'mean',
        'Line width': 'mean',
        'Intensity': 'mean',
        'Contrast': 'mean',
        'Sinuosity': 'mean',
        'Gaps': 'sum'
    }
    # Perform the groupby and aggregation by Sample and Cell
    summary = data.groupby(['Sample', 'Cell']).agg(aggregations).reset_index()
    # Make it so the index is the Sample
    summary = summary.set_index('Sample')
    # Flatten the columns

    # Exclude the selected columns
    if columns_to_exclude:
        summary = summary.drop(columns=columns_to_exclude, errors='ignore')

    # Rename the columns
    summary = summary.rename(columns={
        'Sample': 'Sample',
        'Network': '# Networks',
        'Contour': '# Contours',
        'Length': 'Avg Length',
        'Line width': 'Avg Line Width',
        'Intensity': 'Avg Intensity',
        'Contrast': 'Avg Contrast',
        'Sinuosity': 'Avg Sinuosity',
        'Gaps': '# Gaps'
    })

    # Change Networks, Contours, Intensity, and Gaps to integers. For the rest limit decimal points to 4
    summary['# Networks'] = summary['# Networks'].astype(int)
    summary['# Contours'] = summary['# Contours'].astype(int)
    summary['# Gaps'] = summary['# Gaps'].astype(int)

    # Calculate the ratio of networks and gaps to contours
    summary['Netw/Cont'] = (summary['# Networks'] / summary['# Contours']).round(4)
    summary['Gaps/Cont'] = (summary['# Gaps'] / summary['# Contours']).round(4)

    if type_to_analyze == 'Sample':
        # Drop the 'Cell' column
        summary = summary.drop(columns='Cell')
        # Group by 'Sample' and calculate the mean
        summary = summary.groupby('Sample').mean()

        # Rename the columns networks, contours, and gaps
        summary = summary.rename(columns={
            '# Networks': 'Avg # Networks',
            '# Contours': 'Avg # Contours',
            '# Gaps': 'Avg # Gaps'
        })

        summary['Avg # Networks'] = summary['Avg # Networks'].astype(int)
        summary['Avg # Contours'] = summary['Avg # Contours'].astype(int)
        summary['Avg # Gaps'] = summary['Avg # Gaps'].astype(int)

    
    summary['Avg Intensity'] = summary['Avg Intensity'].astype(int)
    summary['Avg Contrast'] = summary['Avg Contrast'].round(4)
    summary['Avg Sinuosity'] = summary['Avg Sinuosity'].round(4)

    # Change the netw/cont and gaps/cont to percentages with 2 decimal points
    summary['Netw/Cont'] = (summary['Netw/Cont'] * 100).round(2)
    summary['Gaps/Cont'] = (summary['Gaps/Cont'] * 100).round(2)
    # Change name to include percentage
    summary = summary.rename(columns={
        'Netw/Cont': 'Netw/Cont (%)',
        'Gaps/Cont': 'Gaps/Cont (%)'
    })

    #summary.columns = [' '.join(col).strip() for col in summary.columns.values]

    
    return summary


def calculate_summarized_metrics_2(data, type_to_analyze, columns_to_exclude):

    # Define the aggregation methods for each column and add std deviation
    aggregations = {
        'Network': lambda x: x[x != 0].nunique(),
        'Contour': 'count',
        'Length': ['mean', 'std'],
        'Line width': ['mean', 'std'],
        'Intensity': ['mean', 'std'],
        'Contrast': ['mean', 'std'],
        'Sinuosity': ['mean', 'std'],
        'Gaps': 'sum'
    }
    # Perform the groupby and aggregation by Sample and Cell
    summary = data.groupby(['Sample', 'Cell']).agg(aggregations).reset_index()
    # Flatten the multi-level column names
    summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]
    # Make it so the index is the Sample
    summary = summary.set_index('Sample_')
    # Flatten the columns

    # Exclude the selected columns
    if columns_to_exclude:
        summary = summary.drop(columns=columns_to_exclude, errors='ignore')

    # Rename the columns
    summary = summary.rename(columns={
        'Sample': 'Sample',
        'Network_<lambda>': '# Networks',
        'Contour_count': '# Contours',
        'Length_mean': 'Avg Length',
        'Line width_mean': 'Avg Line Width',
        'Intensity_mean': 'Avg Intensity',
        'Contrast_mean': 'Avg Contrast',
        'Sinuosity_mean': 'Avg Sinuosity',
        'Gaps_sum': '# Gaps'
    })

    # Change Networks, Contours, Intensity, and Gaps to integers. For the rest limit decimal points to 4
    summary['# Networks'] = summary['# Networks'].astype(int)
    summary['# Contours'] = summary['# Contours'].astype(int)
    summary['# Gaps'] = summary['# Gaps'].astype(int)

    # Calculate the ratio of networks and gaps to contours
    summary['Netw/Cont'] = (summary['# Networks'] / summary['# Contours']).round(4)
    summary['Gaps/Cont'] = (summary['# Gaps'] / summary['# Contours']).round(4)

    if type_to_analyze == 'Sample':
        # Drop the 'Cell' column
        summary = summary.drop(columns='Cell_')
        # Group by 'Sample' and calculate the mean and std deviation
        summary = summary.groupby('Sample_').agg(['mean', 'std'])

        # Rename the columns networks, contours, and gaps
        #summary = summary.rename(columns={
        #    '# Networks': 'Avg # Networks',
        #    '# Contours': 'Avg # Contours',
        #    '# Gaps': 'Avg # Gaps'
        #})

        #summary['Avg # Networks'] = summary['Avg # Networks'].astype(int)
        #summary['Avg # Contours'] = summary['Avg # Contours'].astype(int)
        #summary['Avg # Gaps'] = summary['Avg # Gaps'].astype(int)

    
    #summary['Avg Intensity'] = summary['Avg Intensity'].astype(int)
    #summary['Avg Contrast'] = summary['Avg Contrast'].round(4)
    #summary['Avg Sinuosity'] = summary['Avg Sinuosity'].round(4)

    # Change the netw/cont and gaps/cont to percentages with 2 decimal points
    #summary['Netw/Cont'] = (summary['Netw/Cont'] * 100).round(2)
    #summary['Gaps/Cont'] = (summary['Gaps/Cont'] * 100).round(2)
    # Change name to include percentage
    #summary = summary.rename(columns={
    #    'Netw/Cont': 'Netw/Cont (%)',
    #    'Gaps/Cont': 'Gaps/Cont (%)'
    #})

    #summary.columns = [' '.join(col).strip() for col in summary.columns.values]

    
    return summary