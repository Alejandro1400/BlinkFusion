from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from Dashboard.Radar import ComplexRadar

def remove_outliers(df, column):
    """
    Remove outliers based on the IQR method.
    """
    Q1 = df[column].quantile(0.30)
    Q3 = df[column].quantile(0.70)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtering the data
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def create_boxplot(data, x_label, y_label, outliers):
    # Apply the appropriate aggregation based on y_label
    if y_label == 'Gaps':
        data = data[['Sample', 'Cell', y_label]]
        data = data.groupby(['Sample', 'Cell']).sum().reset_index()
    elif y_label in ['Network', 'Contour']:
        data = data[['Sample', 'Cell', y_label]]
        data = data[data[y_label] != 0]
        data = data.groupby(['Sample', 'Cell']).nunique().reset_index()
    elif y_label == 'Gaps/Cont':
        data = data[['Sample', 'Cell', 'Gaps', 'Contour']]
        data = data.groupby(['Sample', 'Cell']).agg({'Gaps': 'sum', 'Contour': 'count'}).reset_index()
        data['Gaps/Cont'] = data['Gaps'] / data['Contour']
    elif y_label == 'Netw/Cont':
        data = data[['Sample', 'Cell', 'Network', 'Contour']]
        data = data.groupby(['Sample', 'Cell']).nunique().reset_index()
        data['Netw/Cont'] = data['Network'] / data['Contour']
    
    if outliers == True:
        # Remove outliers
        data = remove_outliers(data, y_label)

    # Sort data
    data = data.sort_values(by=[y_label], ascending=True)
    
    # Set the y-axis label
    y_axis = y_label

    if y_label in ['Netw/Cont', 'Gaps/Cont']:
        # Change to percentage and round to 2 decimal places
        data[y_label] = data[y_label] * 100
        data[y_label] = np.round(data[y_label], 2)
        # Change the x_axis to show percentage
        y_axis = f'{x_label} (%)'

    # Plotting based on the x_label
    if x_label == 'Cell':

        # Grab the first part of the sample name before a space and add to cell name
        data['Cell'] = data['Cell'] + ' (' + data['Sample'].str.split(' ').str[0] + ')'

        # Create a boxplot for each cell within each sample
        fig = px.box(data, x='Sample', y=y_label, color='Cell', 
                     labels={'Cell': 'Cell Name'}, points=False, 
                     title=f'Boxplot of {y_label} by {x_label}')
        # Customize hover data to show the cell name
        fig.update_traces(hoverinfo='y+name', whiskerwidth=0.2)
    else:
        # Normal boxplot for Sample grouping
        fig = px.box(data, x=x_label, y=y_label, points=False,
                     title=f'Boxplot of {y_label} by {x_label}')
        fig.update_traces(hoverinfo='y', whiskerwidth=0.2)

    # Update layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_axis,
        boxmode='group'  # Group boxes together based on x-axis categories
    )
    
    return fig


def create_histogram(data, x_label, bins, title, histtype='step'):
    # Grab only from where Sample corresponds to the title
    data = data[data['Sample'] == title]

    if x_label == 'Netw/Cont':
        # Only keep the columns Sample, Cell, and y_label
        data = data[['Sample', 'Cell', 'Network', 'Contour']]
        data = data.groupby(['Sample', 'Cell']).nunique().reset_index()
        data['Netw/Cont'] = data['Network'] / data['Contour']
    elif x_label == 'Gaps/Cont':
        # Only keep the columns Sample, Cell, and y_label
        data = data[['Sample', 'Cell', 'Gaps', 'Contour']]
        aggregations = {
            'Gaps': 'sum',
            'Contour': 'count'
        }
        data = data.groupby(['Sample', 'Cell']).agg(aggregations).reset_index()
        data['Gaps/Cont'] = data['Gaps'] / data['Contour']
    elif x_label == 'Gaps':
         # Only keep the columns Sample, Cell, and y_label
        data = data[['Sample', 'Cell', x_label]]
        data = data.groupby(['Sample', 'Cell']).sum().reset_index()
    elif x_label in ['Network', 'Contour']:
        # Only keep the columns Sample, Cell, and x_label
        data = data[['Sample', 'Cell', x_label]]
        data = data[data[x_label] != 0]
        data = data.groupby(['Sample', 'Cell']).nunique().reset_index()


    # Extract data for the histogram
    data_values = data[x_label].values

    if x_label in ['Netw/Cont', 'Gaps/Cont']:
        # Change to percentage and round to 2 decimal places
        data_values = data_values * 100
        data_values = np.round(data_values, 2)
        # Change the x_label to show percentage
        x_label = f'{x_label} (%)'

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = ax.hist(data_values, bins=bins, histtype=histtype, color='blue', alpha=0.5, rwidth=0.8)

    # Fit a normal distribution to the data
    #mu, std = norm.fit(data_values)
    
    # Create a range of x values for the normal distribution curve
    #x = np.linspace(min(data_values), max(data_values), 300)
    
    # Calculate the normal distribution values
    #p = norm.pdf(x, mu, std)
    
    # Plotting the normal distribution curve
    #ax.plot(x, p, 'r--', linewidth=2, label=f'Normal Fit: μ={mu:.2f}, σ={std:.2f}')

    # Set titles and labels with font settings
    ax.set_title(f'Histogram of {x_label} for {title}', fontsize=12, fontname='Arial')
    ax.set_xlabel(x_label, fontsize=10, fontname='Arial')
    ax.set_ylabel('Frequency', fontsize=10, fontname='Arial')
    
    # Add legend with the mean and standard deviation
    ax.legend(loc='upper right', fontsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return fig

def calculate_custom_ranges(df, categories):
    ranges = []
    for category in categories:
        max_value = df[category].max()
        min_value = df[category].min()

        if category == 'Sinuosity':
            # Sinuosity's min is 1 and max is the closest 0.1 above the max
            range_min = 1
            range_max = np.ceil(max_value * 10) / 10  # Rounds up to the nearest 0.1
        else:
            if max_value > 100:
                # Round max to the nearest 10
                range_max = np.ceil(max_value / 10) * 10
                increment = 10
            elif max_value > 10:
                # Round max to the nearest 5
                range_max = np.ceil(max_value / 5) * 5
                increment = 5
            else:
                # Round max to the nearest 1
                range_max = np.ceil(max_value)
                increment = 1

            # Calculate half of the max value for minimum range consideration
            half_max = max_value / 2

            if min_value > half_max:
                # If all values are greater than half of the max, start from half of the max
                range_min = np.floor(half_max / increment) * increment
            else:
                # Otherwise, start from zero
                range_min = 0

        # Append calculated ranges
        ranges.append((range_min, range_max))

    return ranges

def create_radar_chart(df, id_column='Sample'):

    # Reset index and rename if necessary
    if not id_column in df.columns:
        df = df.reset_index().rename(columns={'index': id_column})

    # Remove unnecessary columns if they exist
    columns_to_remove = ['Cell', 'Avg # Contours', 'Avg # Networks', 'Avg # Gaps']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

    # Calculate Fill Index (%) as the opposite of Gaps/Cont
    df['Fill Index (%)'] = 100 - df['Gaps/Cont (%)']

    # Remove Gaps/Cont column
    df = df.drop(columns=['Gaps/Cont (%)'], errors='ignore')

    df = df.rename(columns={
        'Avg Length': 'Length',
        'Avg Line Width': 'Line Width',
        'Avg Intensity': 'Intensity',
        'Avg Contrast': 'Contrast',
        'Avg Sinuosity': 'Sinuosity',
        'Netw/Cont (%)': 'Network Ratio (%)'
    })

    # Identify and retain numeric columns only for plotting
    categories = df.select_dtypes(include=[np.number]).columns.tolist()

    # Obtain the ranges for each category, which are the min value to max value
    ranges = calculate_custom_ranges(df, categories)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))  # Adjusted smaller size here
    radar = ComplexRadar(fig, categories, ranges)
    
    for _, row in df.iterrows():
        data = row[categories].tolist()
        radar.plot(data, label=row[id_column])
    
    return fig