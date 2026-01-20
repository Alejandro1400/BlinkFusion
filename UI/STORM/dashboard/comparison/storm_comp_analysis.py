import streamlit as st 
import plotly.express as px
import plotly.graph_objs as go

def comparison_analysis(metridata, desc_columns, metrics_columns):
    """
    Allows users to compare metrics using custom plots and visualizations.
    Args:
        metridata (DataFrame): DataFrame containing merged metric and metadata.
        desc_columns (list): List of descriptive columns from metadata.
        metrics (DataFrame): DataFrame containing metrics data.
    """
    st.subheader("Metrics Comparison")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_axis = st.selectbox(
            'X-Axis:',
            options=list(desc_columns),
            key='x_select',
            help="Select a column for the X-axis of the plot."
        )

    with col2:
        # Set of columns to remove
        columns_to_remove = {'Experiment ID', 'QE Period (s)', '# Images'}

        # Create a new list excluding the columns in columns_to_remove using set for faster lookup
        filtered_columns = [col for col in metrics_columns if col not in columns_to_remove]
        y_axis = st.selectbox(
            'Y-Axis:',
            options=list(filtered_columns),
            key='y_select',
            help="Select a column for the Y-axis of the plot."
        )

    with col3:
        available_legends = [col for col in desc_columns if col != 'IDENTIFIER' and col != x_axis]
        legend_column = st.selectbox(
            'Legend (optional):',
            options=['None'] + available_legends,
            key='legend_select',
            help="Choose a column to group data by legend (optional)."
        )

    with col4:
        plot_type = st.selectbox(
            'Plot Type:',
            options=['Bar', 'Line', 'Box', 'Violin'],
            key='plot_type_select',
            help="Select the type of plot to visualize data."
        )

    if not metridata.empty:
        if legend_column != 'None':
            grouped_metrics = metridata.groupby([x_axis, legend_column])[y_axis].mean().reset_index()
        else:
            grouped_metrics = metridata.groupby(x_axis)[y_axis].mean().reset_index()

        plot_kwargs = {'x': x_axis, 'y': y_axis, 'height': 400}
        if legend_column != 'None':
            plot_kwargs['color'] = legend_column

        if plot_type == 'Bar':
            fig = px.bar(grouped_metrics, **plot_kwargs)
        elif plot_type == 'Line':
            fig = px.line(grouped_metrics, **plot_kwargs)
        elif plot_type == 'Box':
            fig = px.box(metridata, **plot_kwargs)
        elif plot_type == 'Violin':
            fig = px.violin(metridata, **plot_kwargs)

        st.plotly_chart(fig, use_container_width=True)
        display_columns = [x_axis, y_axis] + ([legend_column] if legend_column != 'None' else [])
        display_df = grouped_metrics.set_index(x_axis)
        st.dataframe(display_df, use_container_width=True, height=200)


# def time_series_comparison(timeseries, metadata_analysis):
#     """
#     Allows users to configure and view time-series comparisons.
#     Args:
#         timeseries (DataFrame): DataFrame containing time-series data.
#         timeseries_analysis (DataFrame): DataFrame with analysis results.
#         metadata_analysis (DataFrame): DataFrame with metadata.
#     """
#     st.markdown("___")
#     st.subheader("Time Series Comparison")
#     col1, col2, col3 = st.columns(3)
#     st.write(timeseries)
#     for key, value in timeseries.items():
#         print(f"{key}: {type(value)}")

#     time_series_columns = timeseries.columns.drop('IDENTIFIER')

#     with col1:
#         num_axes = st.selectbox(
#             'Number of Y-Axes:',
#             options=[1, 2],
#             index=0,
#             key='num_axes_select',
#             help="Select the number of Y-axes to display."
#         )

#     with col2:
#         y1_label = st.selectbox(
#             'Y1 Axis:',
#             options=list(time_series_columns),
#             key='y1_select',
#             help="Select a column for the primary Y-axis."
#         )

#     y2_label = None
#     if num_axes == 2:
#         with col3:
#             y2_label = st.selectbox(
#                 'Y2 Axis:',
#                 options=list(time_series_columns),
#                 key='y2_select',
#                 help="Select a column for the secondary Y-axis."
#             )

#     timeseries_analysis = timeseries_analysis.reset_index()
#     timeseries_data = timeseries_analysis.merge(metadata_analysis, on='IDENTIFIER', how='inner').set_index('index')

#     if not timeseries_data.empty:
#         fig = go.Figure()

#         grouped_y1 = timeseries_data[y1_label].groupby(timeseries_data.index).mean()
#         fig.add_trace(go.Scatter(x=grouped_y1.index, y=grouped_y1, mode='lines', name=y1_label))

#         if y2_label:
#             grouped_y2 = timeseries_data[y2_label].groupby(timeseries_data.index).mean()
#             fig.add_trace(go.Scatter(x=grouped_y2.index, y=grouped_y2, mode='lines', name=y2_label, yaxis='y2'))

#             fig.update_layout(
#                 yaxis2=dict(
#                     title=y2_label,
#                     overlaying='y',
#                     side='right'
#                 )
#             )

#         fig.update_layout(
#             title="Time Series Comparison",
#             xaxis_title="Time (s)",
#             yaxis_title=y1_label,
#             hovermode="x"
#         )

#         st.plotly_chart(fig, use_container_width=True)
#         display_columns = ['Time (s)', y1_label] + ([y2_label] if y2_label else [])
#         display_df = timeseries_data.reset_index().rename(columns={'index': 'Time (s)'})[display_columns]
#         st.dataframe(display_df, use_container_width=True, height=200)