import numpy as np
import pandas as pd
import plotly.graph_objects as go
from package.agent import Agent
from plotly.subplots import make_subplots

# from tqdm.auto import tqdm


# flake8: noqa


class Q_learning_Agent(Agent):
    def __init__(
        self, gamma: float = 1, step_size: float = 0.1, epsilon: float = 0.1
    ) -> None:
        super().__init__(gamma, step_size, epsilon)
        self.name = "Q-learning"
        # used to display the performances of the model for every step
        self.cumulative_rewards = []
        self.reward_counter = 0

    def agent_start(self, state: int):
        """
        Called at the start of an episode, takes the first action
        given the initial state
        """
        self.past_state = state
        self.past_action = self.epsilon_greedy(state)
        # take the action
        self.update_state(state, self.past_action)
        return self.past_action

    def step(self, state: int, reward: int) -> None:
        # direct RL update
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * (
            reward + self.gamma * np.max(self.q_values[state]) - update
        )
        self.q_values[self.past_state][self.past_action] = update
        # action selection using the e-greedy policy
        action = self.epsilon_greedy(state)
        self.update_state(state, action)
        # before performing the action, save the current state and action
        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_end(self) -> None:
        """
        Called once the agent reaches a terminal state
        """
        terminal_coordinates = self.state_to_coord(self.position)
        # the coordinates must be reversed when querying the dataframe
        reward = self.env.get_reward(terminal_coordinates)
        # direct RL update for a terminal state
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * (reward - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update with next_action = -1
        self.state_visits[self.past_state] += 1

    def play_episode(self) -> None:
        """
        Plays one episode (agent_start, agent_step, agent_end)
        Records the number of step during the episode
        """
        self.agent_start(self.start_position)
        episode_steps = 1
        while not self.done:
            reward = self.env.get_reward(self.state_to_coord(self.position))
            self.step(self.position, reward)
            episode_steps += 1
        self.n_steps.append(episode_steps)
        self.agent_end()
        self.reset()

    def fit(
        self, n_episode: int, log_progress: list = None, plot: bool = False
    ) -> None:
        """
        Plays n_episode episodes
        @log_progress (list): calls the log_agent_performances
        function for each episode in the list
        @plot (bool): whether to plot the recorded progress or not
        """
        self.value_estimates = {}
        self.episode_played = 0
        for idx in range(n_episode):
            if self.episode_played == 100:
                self.env.activate_late_portal()
            self.play_episode()
            self.value_estimates[self.episode_played] = self.state_dict_to_matrix(
                self.q_values
            ).values

            self.episode_played += 1
            if log_progress is not None:
                if idx in log_progress:
                    self.log_agent_performances(plot=plot)

    def state_dict_to_matrix(self, dictionary: dict) -> pd.DataFrame:
        """
        Convert a dictionary of states (e.g. q_values and n_visits) to a
        matrix representation matching the environment's grid representation
        """
        key_val = [
            (self.state_to_coord(key)[::-1], np.max(values))
            for (key, values) in list(dictionary.items())
        ]
        matrix = pd.DataFrame(np.zeros((8, 12)))
        for key, value in key_val:
            matrix.loc[key] = value
        matrix.index = [str(i) for i in matrix.index]
        matrix.columns = [str(c) for c in matrix.columns]
        return matrix

    def plot_heatmap(self, matrix, **kwargs):
        return go.Heatmap(
            z=matrix.values[::-1],
            x=matrix.columns,
            y=matrix.index[::-1],
            colorscale="Plasma",
            **kwargs,
        )

    def plot_bar_chart(self, dataframe: pd.DataFrame, attribute: "str", color: str):
        return go.Bar(y=dataframe[attribute], marker=dict(color=dataframe[color]))

    def log_agent_performances(self, plot: bool = False) -> None:
        """
        Records the steps and reward for each episode until the current time step
        @plot: plots the current state of the agent's q values, the number of visits for each state
        and the number of steps per episode
        """
        q_values = self.state_dict_to_matrix(self.q_values)
        state_visits = self.state_dict_to_matrix(self.state_visits)

        # # assign the color green if the agent finds the treasure within 12 steps, red if 17 steps
        # # blue if the agent terminates before finding the reward or in more than 17 steps
        episodes = pd.DataFrame({"steps": self.n_steps, "reward": self.rewards})
        episodes["is_optimal"] = "#636EFA"  # Assign default blue color
        episodes.loc[
            (episodes["reward"] == 1) & (episodes["steps"].between(13, 17)),
            "is_optimal",
        ] = "#EF553B"  # Red color
        episodes.loc[
            (episodes["reward"] == 1) & (episodes["steps"] <= 12), "is_optimal"
        ] = "#00CC96"  # Green color
        episodes["is_optimal"] = pd.Categorical(episodes["is_optimal"])
        self.episodes = episodes

        if plot:
            # create the plots
            heatmap1 = self.plot_heatmap(
                q_values, **{"colorbar": dict(x=0.45, y=0.78, len=0.473)}
            )
            heatmap2 = self.plot_heatmap(
                state_visits, **{"colorbar": dict(x=1, y=0.78, len=0.473)}
            )
            bar_chart = self.plot_bar_chart(
                episodes.tail(100), attribute="steps", color="is_optimal"
            )

            # Create subplot figure
            fig = make_subplots(
                rows=2,
                cols=2,
                shared_xaxes=False,
                vertical_spacing=0.13,
                specs=[[{}, {}], [{"colspan": 2}, None]],
                subplot_titles=(
                    "State value function",
                    "Number of total visits",
                    "Number of steps per episode (last 100 steps) green bar = 12 steps, red bars = 17 or less steps",
                ),
            )

            # Add the heatmaps and bar chart to the subplot figure
            fig.add_trace(heatmap1, row=1, col=1)
            fig.add_trace(heatmap2, row=1, col=2)
            fig.add_trace(bar_chart, row=2, col=1)

            title = f"Agent: {self.name}, Number of episodes: {self.episode_played}<br>\
            <span style='font-size: 13px'>Model parameters: [learning rate: {self.step_size}, epsilon: {self.epsilon}, discount: {self.gamma}]</span>"
            fig.update_layout(height=900, width=1200, title=title)

            fig.show()
