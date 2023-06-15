from package import Dyna_Q_Agent
import numpy as np


class Dyna_Q_plus_Agent(Dyna_Q_Agent):
    """
    In Dyna-Q+, a bonus reward is given for actions that haven't been tried for a long time
    as there are greater chances that the environment dynamics have changed
    The number of transitions since the last time (state, action)
    was tried is given by tau(state, action)
    The associated reward is given by: reward + kappa * sqrt(tau(state, action))
    """
    def __init__(self,
                 gamma: float = 1,
                 step_size: float = 0.1,
                 epsilon: float = 0.1,
                 planning_steps: int = 100,
                 kappa: float = 1e-3,
                 ) -> None:
        super().__init__(gamma, step_size, epsilon, planning_steps)
        self.name = "Dyna-Q+"
        self.kappa = kappa
        self.tau = self.init_state_action_dict()

    def update_model(self, last_state: int, last_action: int, state: int, reward: int) -> None:
        """
        Overwrite the Dyna-Q update_model function
        Now, when we visit a state for the first time, all the action that were not selected
        are initialized with 0, they will be updated at each time steps
        according to the Dyna-Q+ algorithm
        """
        if last_state not in self.model:
            self.model[last_state] = {last_action: (state, reward)}
        for action in self.actions:
            if action != last_action:
                self.model[last_state][action] = (last_state, 0)
        else:
            self.model[last_state][last_action] = (state, reward)

    def update_tau(self, state: int, action: int) -> None:
        for key in list(self.tau.keys()):
            self.tau[key] += 1
        self.tau[state][action] = 0

    def planning_step(self) -> None:
        """
        Overwrite the Dyna-Q planning_step function
        Performs planning (indirect RL) and adds a bonus to the transition reward
        The bonus is given by kappa * sqrt(tau(state, action))
        """
        for _ in range(self.planning_steps):
            # select a visited state
            planning_state = self.random_generator.choice(list(self.model.keys()))
            # select a recorded action
            planning_action = self.random_generator.choice(
                list(self.model[planning_state].keys()))
            # get the predicted next state and reward
            next_state, reward = self.model[planning_state][planning_action]
            # add the bonus reward
            reward += self.kappa * np.sqrt(self.tau[planning_state][planning_action])
            # update the values in case of terminal state
            if next_state == -1:
                update = self.q_values[planning_state][planning_action]
                update += self.step_size * (reward - update)
                self.q_values[planning_state][planning_action] = update
            # update the values in case of non-terminal state
            else:
                update = self.q_values[planning_state][planning_action]
                update += self.step_size * (reward + self.gamma
                                            * np.max(self.q_values[next_state]) - update)
                self.q_values[planning_state][planning_action] = update

    def step(self, state: int, reward: int) -> None:
        """
        Overwrite the Dyna-Q step function
        At every step, we increment the last visit counter for every state action by 1
        The current state action pair is reset to 0
        """
        # direct RL update
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * (reward + self.gamma * np.max(self.q_values[state]) - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update
        self.update_model(self.past_state, self.past_action, state, reward)
        # planning step
        self.planning_step()
        # action selection using the e-greedy policy
        action = self.epsilon_greedy(state)
        self.update_tau(state, action)
        self.update_state(state, action)
        # before performing the action, save the current state and action
        self.past_state = state
        self.past_action = action

        return self.past_action
