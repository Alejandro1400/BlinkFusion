from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    #data = data[data['Sample'] == title]

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


def plot_time_series_interactive(metric1_data, metric2_data, metric1_name, metric2_name, qe_start, qe_end, qe_dc, qe_sf):
    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(x=metric1_data.index, y=metric1_data, name=metric1_name, mode='lines', line=dict(color='red'), yaxis='y', showlegend=False)
    )

    fig.add_trace(
        go.Scatter(x=metric2_data.index, y=metric2_data, name=metric2_name, mode='lines', line=dict(color='blue', dash='dot'), yaxis='y2', showlegend=False)
    )

    # Add a shaded area (transparent rectangle) from qe_start to qe_end
    fig.add_shape(
        type="rect",
        x0=qe_start,
        x1=qe_end,
        y0=0,
        y1=1,  # This is relative to the y-axis range of 'y' (Metric 1)
        yref="paper",  # Uses the full vertical range of the plot
        fillcolor="lightgray",
        opacity=0.5,
        layer="below",
        line_width=0,

    )

    # Add annotation for Metric 1 and Metric 2 in the middle of the shaded area
    midpoint_x = (qe_start + qe_end) / 2
    fig.add_annotation(
        x=midpoint_x,
        y=0.6,  # Position near the top of the plot for visibility
        yref="paper",
        text=f"Duty Cycle: {qe_dc:.5f}<br>Survival Fraction: {qe_sf*100:.1f}%",  # Metric 1 to 5 decimals, Metric 2 as percentage with 1 decimal
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center",
        opacity=0.8
    )

    # Create axis objects
    fig.update_layout(
        xaxis=dict(title='Time (s)', showgrid=False, titlefont=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title=metric1_name, titlefont=dict(color='black'), tickfont=dict(color='red'), range=[0, max(metric1_data)], showgrid=False),
        yaxis2=dict(title=metric2_name, titlefont=dict(color='black'), tickfont=dict(color='blue'), overlaying='y', side='right', range=[0, max(metric2_data)], showgrid=False),
        title=f'{metric1_name} and {metric2_name} Over Time'
    )

    # Update layout to be more aesthetically pleasing
    fig.update_layout(showlegend=True, plot_bgcolor='white', paper_bgcolor='white')

    return fig




def plot_intensity_vs_frame(plot_data):
    time = plot_data.index * 50 / 1000  # Convert frame to time in seconds
    # Create a Plotly figure
    fig = go.Figure()

    # Add a fill area trace
    fig.add_trace(go.Scatter(
        x=time,  # Use the time variable for the x-axis
        y=plot_data['INTENSITY'],
        fill='tozeroy',  # Fill to zero on the y-axis
        mode='lines',  # Line plot
        line_color='black',  # Line color
        line_shape='hv'  # Horizontal and vertical steps
    ))

    # Customize layout
    fig.update_layout(
        title='Intensity vs Frame',
        xaxis=dict(title='Time (s)', range=[0, 500]),
        xaxis_title='Time (s)',
        yaxis_title='Intensity',
        template='plotly_white',  # White background template
        showlegend=False  # Do not show legend
    )

    return fig

def remove_outliers_upper(data, num_bins):
    # Generate histogram data
    counts, bin_edges = np.histogram(data, num_bins)

    # Traverse the counts in reverse to find the threshold bin
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] > 3:  # Keep bins with more than 1 count
            threshold = bin_edges[i + 1]  # Upper threshold (next bin edge)
            break

    # Filter the data based on the threshold
    filtered_data = [x for x in data if x <= threshold]
    return filtered_data


def plot_histograms(duty_cycle, photons, switching_cycles, on_time, metrics, remove_outliers=False, num_bins=40):
    # Remove outliers if specified
    # Note: This is a simple method to remove outliers based on the upper threshold
    if remove_outliers:
        duty_cycle = remove_outliers_upper(duty_cycle, num_bins)
        photons = remove_outliers_upper(photons, num_bins)
        switching_cycles = remove_outliers_upper(switching_cycles, num_bins)
        on_time = remove_outliers_upper(on_time, num_bins)

    # Convert to x10^3 for better readability of the photons
    photons = [x / 1000 for x in photons]

    # Grab the first row of the metrics DataFrame
    metrics = metrics.iloc[0]

    # Calculate the mean of the duty_cycle series for QE Duty Cycle
    duty_cycle_mean = np.mean(duty_cycle) if len(duty_cycle) > 0 else "N/A"
    switching_cycles_mean = np.mean(switching_cycles) if len(switching_cycles) > 0 else "N/A"
    on_time_mean = np.mean(on_time) if len(on_time) > 0 else "N/A"
    photons_mean = np.mean(photons) if len(photons) > 0 else "N/A"


    # Extract relevant metrics from the metrics dataset
    duty_cycle_metrics = {
        "QE Duty Cycle": round(duty_cycle_mean, 5) if duty_cycle_mean != "N/A" else "N/A",
        "QE Survival Fraction": metrics.get("QE Survival Fraction", "N/A").round(3)
    }
    switching_cycles_metrics = {
        "SC per Mol": metrics.get("SC per Mol", "N/A").round(2),
        "QE SC per Mol": round(switching_cycles_mean, 2) if switching_cycles_mean != "N/A" else "N/A"
    }
    on_time_metrics = {
        "On Time per SC (s)": metrics.get("On Time per SC (s)", "N/A").round(2),
        "QE On Time per SC (s)": round(on_time_mean, 2) if on_time_mean != "N/A" else "N/A"
    }
    photons_metrics = {
        "Int. per SC (Photons)": metrics.get("Int. per SC (Photons)", "N/A").round(2),
        "QE Int. per SC (Photons)": round(photons_mean, 2) if photons_mean != "N/A" else "N/A"
    }

    def add_annotations(fig, metric_dict, bg_color):
        text = "<br>".join([f"{key}: {value}" for key, value in metric_dict.items()])
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1, y=1,  # Top-right corner
            text=text,
            showarrow=False,
            font=dict(size=16, color="black"),  # Font size increased
            align="right",
            bgcolor=bg_color,  # Dynamic background color
            bordercolor="black",
            borderwidth=1
        )

    # Create Plotly Histograms with log-scaled x-axis
    duty_cycle_fig = go.Figure(data=[go.Histogram(x=duty_cycle, nbinsx=num_bins, marker_color='lightblue')])
    duty_cycle_fig.update_layout(
        title='Duty Cycle per Molecule',
        xaxis=dict(title='Duty Cycle'),
        yaxis=dict(title='Frequency', type='log'),
        bargap=0.2
    )
    add_annotations(duty_cycle_fig, duty_cycle_metrics, "lightblue")  # Background matches bar color

    switching_cycles_fig = go.Figure(data=[go.Histogram(x=switching_cycles, nbinsx=num_bins, marker_color='lightcoral')])
    switching_cycles_fig.update_layout(
        title='Switching Cycles per Molecule',
        xaxis=dict(title='Switching Cycles'),
        yaxis=dict(title='Frequency', type='log'),
        bargap=0.2
    )
    add_annotations(switching_cycles_fig, switching_cycles_metrics, "lightcoral")  # Background matches bar color

    on_time_fig = go.Figure(data=[go.Histogram(x=on_time, nbinsx=num_bins, marker_color='lightgreen')])
    on_time_fig.update_layout(
        title='On Time per Molecule',
        xaxis=dict(title='On Time (s)'),
        yaxis=dict(title='Frequency', type='log'),
        bargap=0.2
    )
    add_annotations(on_time_fig, on_time_metrics, "lightgreen")  # Background matches bar color

    photons_fig = go.Figure(data=[go.Histogram(x=photons, nbinsx=num_bins, marker_color='yellow')])
    photons_fig.update_layout(
        title='Photons per Molecule',
        xaxis=dict(title='Photons (x10^3)'),
        yaxis=dict(title='Frequency', type='log'),
        bargap=0.2
    )
    add_annotations(photons_fig, photons_metrics, "yellow")  # Background matches bar color


    # Using columns to create a 2x2 grid layout in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(duty_cycle_fig, use_container_width=True)
        st.plotly_chart(on_time_fig, use_container_width=True)

    with col2:
        st.plotly_chart(switching_cycles_fig, use_container_width=True)
        st.plotly_chart(photons_fig, use_container_width=True)