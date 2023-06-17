import os

from package.dyna_q_agent import Dyna_Q_Agent
from package.dyna_q_plus_agent import Dyna_Q_plus_Agent
from package.plots import (
    animated_heatmap,
    plot_average_cumulative_reward,
    plot_steps_per_episode,
)
from package.q_learning_agent import Q_learning_Agent

# if __name__ == "__main__":
#     agents = {
#         "Q-learning": Q_learning_Agent(epsilon=0.1, gamma=0.9, step_size=0.25),
#         "Dyna-Q": Dyna_Q_Agent(
#             planning_steps=100, epsilon=0.1, gamma=0.9, step_size=0.25
#         ),
#         "Dyna-Q+": Dyna_Q_plus_Agent(
#             planning_steps=100, epsilon=0.1, gamma=0.9, step_size=0.25
#         ),
#     }

#     for agent in agents.values():
#         agent.fit(n_episode=401, plot_progress=[400])
#         animated_heatmap(agent.value_estimates, agent_name=agent.name)

#     plot_steps_per_episode(agents["Q-learning"], agents["Dyna-Q"], agents["Dyna-Q+"])
#     plot_average_cumulative_reward(
#         agents["Q-learning"], agents["Dyna-Q"], agents["Dyna-Q+"]
#     )


if __name__ == "__main__":
    agents = {
        "Q-learning": Q_learning_Agent(epsilon=0.1, gamma=0.9, step_size=0.25),
        "Dyna-Q": Dyna_Q_Agent(
            planning_steps=100, epsilon=0.1, gamma=0.9, step_size=0.25
        ),
        "Dyna-Q+": Dyna_Q_plus_Agent(
            planning_steps=100, epsilon=0.1, gamma=0.9, step_size=0.25
        ),
    }

    n_episodes = 21

    for agent in agents.values():
        agent.fit(n_episode=n_episodes, log_progress=[n_episodes - 1], plot=False)
        animated_heatmap(agent.value_estimates, agent_name=agent.name)

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Write agent's episode results to a CSV file
        results_file = f"results/{agent.name}_results.csv"
        agent.episodes.to_csv(results_file)

    plot_steps_per_episode(agents["Q-learning"], agents["Dyna-Q"], agents["Dyna-Q+"])
    plot_average_cumulative_reward(
        agents["Q-learning"], agents["Dyna-Q"], agents["Dyna-Q+"]
    )
