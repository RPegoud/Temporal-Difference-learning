import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from package.dyna_q_agent import Dyna_Q_Agent
from package.dyna_q_plus_agent import Dyna_Q_plus_Agent
from package.plots import plot_average_reward, plot_steps_per_episode
from package.q_learning_agent import Q_learning_Agent

if __name__ == "__main__":
    agents_parameters = {   
        Q_learning_Agent: {
            "epsilon": 0.1,
            "gamma": 0.9,
            "step_size": 0.25,
        },
        Dyna_Q_Agent: {
            "planning_steps": 100,
            "epsilon": 0.1,
            "gamma": 0.9,
            "step_size": 0.25,
        },
        Dyna_Q_plus_Agent: {
            "planning_steps": 100,
            "epsilon": 0.1,
            "gamma": 0.9,
            "step_size": 0.25,
        },
    }

    num_runs = 10
    num_episodes = 250
    random_seeds = np.arange(num_runs) + 100  # avoid seed 17, only used for testing

    for agent_class, agent_parameters in agents_parameters.items():
        print(agent_class().name)
        agent_results = []

        for run in tqdm(range(num_runs), position=0, leave=True):
            # instantiate a new agent for each run
            agent = agent_class(**agent_parameters)
            # Set a different random seed for each run
            agent.random_generator = np.random.RandomState(seed=random_seeds[run])
            agent.fit(
                n_episode=num_episodes, log_progress=[num_episodes - 1], plot=False
            )

            # Append the episode results to the agent's results list
            agent_results.append(agent.episodes)

        # Concatenate the episode results from all runs
        agent_results_concatenated = pd.concat(
            agent_results, keys=range(num_runs), names=["Run", "episode"]
        )

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Write the concatenated results to a CSV file for all runs
        results_file = f"results/{agent.name}_results.csv"
        agent_results_concatenated.to_csv(
            results_file,
            header=["steps", "reward", "is_optimal"],
            index_label=["Run", "episode"],
        )

    # Compare the agents performances
    plot_average_reward(Q_learning_Agent(), Dyna_Q_Agent(), Dyna_Q_plus_Agent())
    plot_steps_per_episode(Q_learning_Agent(), Dyna_Q_Agent(), Dyna_Q_plus_Agent())
