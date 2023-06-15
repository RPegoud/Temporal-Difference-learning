import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_heatmap(matrix, **kwargs):
    return go.Heatmap(z=matrix.values[::-1],
                      x=matrix.columns,
                      y=matrix.index[::-1],
                      colorscale='Viridis',
                      **kwargs)


def plot_bar_chart(dataframe: pd.DataFrame, attribute: 'str', color: str):
    return go.Bar(y=dataframe[attribute],
                  marker=dict(color=dataframe[color]))


def animated_heatmap(state_value_dict: dict, agent_name: str = None):
    value_estimates_array = np.array(list(state_value_dict.values()))

    fig = go.Figure(
        data=[go.Heatmap(z=value_estimates_array[0])],
        layout=go.Layout(
            title="Frame 0",
            title_x=0.5,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None]
                ),
                 dict(
                    label="Pause",
                    method="animate",
                    args=[None,
                          {
                              "frame": {"duration": 0, "redraw": False},
                              "mode": "immediate",
                              "transition": {"duration": 0}
                          }
                          ]
                )]
            )]
        ),
        frames=[go.Frame(
            data=[go.Heatmap(z=value_estimates_array[i])],
            layout=go.Layout(title_text=f"{agent_name}: State values for episode: {i}"),
            # Decrease the duration for faster transitions (in milliseconds)
            traces=[0],
            name=f"frame_{i}"
        ) for i in range(len(value_estimates_array))]
    )

    fig.update_layout(
        title=f"{agent_name}: State values after episode 1",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        yaxis=dict(autorange="reversed"),
        height=600,
        width=800,
        showlegend=False
    )

    # Decrease the duration for faster transitions (in milliseconds)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 50, "redraw": True},
                                     "fromcurrent": True}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}}]
                    )
                ]
            )
        ]
    )

    fig.show()
