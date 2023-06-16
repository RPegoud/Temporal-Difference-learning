# from package.dyna_q_plus_agent import Dyna_Q_plus_Agent

from package.dyna_q_agent import Dyna_Q_Agent
from package.plots import animated_heatmap

# from package.q_learning_agent import Q_learning_Agent

if __name__ == "__main__":
    a = Dyna_Q_Agent(planning_steps=100, epsilon=0.1, gamma=0.9, step_size=0.5)
    a.fit(n_episode=401, plot_progress=[400])
    animated_heatmap(a.value_estimates, agent_name=a.name)
