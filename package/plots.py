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


def plot_average_reward(*agents):
    fig = go.Figure()

    for agent in agents:
        results_file = f"results/{agent.name}_results.csv"
        agent_results_concatenated = pd.read_csv(
            results_file, index_col=["Run", "episode"]
        )

        # Calculate the average results by episode across the runs
        num_runs = agent_results_concatenated.index.get_level_values("Run").nunique()
        agent_results_average = agent_results_concatenated.groupby("episode").agg(
            "mean", numeric_only=True
        )

        # cumulative_rewards = agent_results_average["reward"].cumsum()
        average_cumulative_reward = (
            agent_results_average / (num_runs + 1) * 100
        )  # Add 1 to include the current run
        fig.add_trace(
            go.Scatter(
                x=average_cumulative_reward.index,
                y=average_cumulative_reward.reward,
                mode="lines",
                name=agent.name,
            )
        )

    fig.update_layout(
        title="Average Reward per Episode (Averaged over 100 runs)",
        xaxis_title="Episode",
        yaxis_title="Average Cumulative Reward",
    )
    fig.show()


def plot_steps_per_episode(*agents):
    fig = go.Figure()

    for agent in agents:
        results_file = f"results/{agent.name}_results.csv"
        agent_results_concatenated = pd.read_csv(
            results_file, index_col=["Run", "episode"]
        )

        # Calculate the average results by episode across the runs
        agent_results_average = agent_results_concatenated.groupby("episode").agg(
            "mean", numeric_only=True
        )

        steps_per_episode = agent_results_average["steps"]

        fig.add_trace(
            go.Scatter(
                x=steps_per_episode.index,
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
        # yaxis_type="log",  # Set y-axis to log scale
    )

    fig.add_shape(
        type="line",
        x0=100,
        x1=100,
        y0=1,
        y1=300,
        line=dict(color="purple", width=1, dash="dot"),
    )

    fig.show()
