import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_heatmap(matrix, **kwargs):
    return go.Heatmap(
        z=matrix.values[::-1],
        x=matrix.columns,
        y=matrix.index[::-1],
        colorscale="Viridis",
        **kwargs,
    )


def plot_bar_chart(dataframe: pd.DataFrame, attribute: "str", color: str):
    return go.Bar(y=dataframe[attribute], marker=dict(color=dataframe[color]))


def animated_heatmap(state_value_dict: dict, agent_name: str = None):
    value_estimates_array = np.array(list(state_value_dict.values()))

    fig = go.Figure(
        data=[go.Heatmap(z=value_estimates_array[0])],
        layout=go.Layout(
            title="Frame 0",
            title_x=0.5,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", args=[None]),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        ),
        frames=[
            go.Frame(
                data=[go.Heatmap(z=value_estimates_array[i])],
                layout=go.Layout(
                    title_text=f"{agent_name}: State values for episode: {i}"
                ),
                traces=[0],
                name=f"frame_{i}",
            )
            for i in range(len(value_estimates_array))
        ],
    )

    fig.update_layout(
        title=f"{agent_name}: State values after episode 1",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        yaxis=dict(autorange="reversed"),
        height=600,
        width=800,
        showlegend=False,
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
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}}],
                    ),
                ],
            )
        ]
    )

    fig.show()


def plot_average_cumulative_reward(*agents):
    fig = go.Figure()

    for agent in agents:
        df = agent.episodes
        df["cumulative_reward"] = df["reward"].cumsum()
        average_cumulative_reward = df["cumulative_reward"] / (df.index + 1)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=average_cumulative_reward, mode="lines", name=agent.name
            )
        )

    fig.update_layout(
        title="Average Cumulative Reward per Episode",
        xaxis_title="Episode",
        yaxis_title="Average Cumulative Reward",
    )
    fig.show()


def plot_steps_per_episode(*agents):
    fig = go.Figure()

    for agent in agents:
        steps_per_episode = agent.episodes["steps"]
        fig.add_trace(
            go.Scatter(
                x=agent.episodes.index,
                y=steps_per_episode,
                name=agent.name,
                mode="lines",
            )
        )

    fig.update_layout(
        title="Steps per Episode",
        xaxis_title="Episode",
        yaxis_title="Steps",
        legend_title="Agent",
        yaxis_type="log",  # Set y-axis to log scale
    )

    fig.add_shape(
        type="line",
        x0=100,
        x1=100,
        y0=1,
        y1=10**5,
        line=dict(color="purple", width=1, dash="dot"),
    )

    fig.update_layout(
        annotations=[
            dict(
                x=100,
                y=10**4,
                xref="x",
                yref="y",
                text="Purple Portal Appeared",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
            )
        ]
    )

    fig.show()
