from package import Q_learning_Agent
import numpy as np


class Dyna_Q_Agent(Q_learning_Agent):
    def __init__(self,
                 gamma: float = 1,
                 step_size: float = 0.1,
                 epsilon: float = 0.1,
                 planning_steps: int = 100) -> None:
        super().__init__(gamma, step_size, epsilon)
        self.planning_steps = planning_steps
        self.model = {}  # model[state][action] = (new state, reward)
        self.name = "Dyna Q"

    def update_model(self, last_state: int, last_action: int, state: int, reward: int) -> None:
        """
        Adds a new transition to the model, if the state is encountered for
        the first time, creates a new key
        """
        try:
            self.model[last_state][last_action] = (state, reward)
            self.state_visits[last_state] += 1
        except KeyError:
            self.model[last_state] = {}
            self.model[last_state][last_action] = (state, reward)

    def planning_step(self) -> None:
        """
        Performs planning (indirect RL)
        """
        for _ in range(self.planning_steps):
            # select a visited state
            planning_state = self.random_generator.choice(list(self.model.keys()))
            # select a recorded action
            planning_action = self.random_generator.choice(
                list(self.model[planning_state].keys()))
            # get the predicted next state and reward
            next_state, reward = self.model[planning_state][planning_action]
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
        A step performed by the agent
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
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.state_visits[self.past_state] += 1
        # planning step
        self.planning_step()
