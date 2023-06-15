from package.agent import Agent
from package.dyna_q_agent import Dyna_Q_Agent
import numpy as np


def test_agent_update_model():
    a = Dyna_Q_Agent()
    a.update_model(6, 1, 7, 0)
    a.update_model(11, 3, 10, 0)
    a.update_model(111, 0, 110, 1)

    assert a.model == {6: {1: (7, 0)},
                       11: {3: (10, 0)},
                       111: {0: (110, 1)}}


def test_argmax():
    """
    Tests the random tie breaking and the reproducibility
    of the experiments
    """
    a = Agent()
    q_values = [0, 2, 3, 4, 1, 2, 4, 4, 3, 4]
    selection = [a.argmax(q_values) for _ in range(10)]
    assert selection == [9, 6, 9, 7, 7, 6, 3, 6, 7, 9]


def test_coord_to_state():
    a = Agent()
    rows, cols = a.env.grid.index, a.env.grid.columns
    for row in rows:
        for col in cols:
            assert a.coord_to_state((col, row)) == col*10 + row


def test_state_coord_identity():
    a = Agent()
    rows, cols = a.env.grid.index, a.env.grid.columns
    for row in rows:
        for col in cols:
            assert a.state_to_coord(a.coord_to_state((col, row))) == (col, row)


def test_update_state_movement():
    a = Agent()
    # attempt to leave the grid from the bottom left corner
    assert a.update_state(7, 3) == 7
    assert a.update_state(7, 2) == 7
    # attempt to leave the grid from the top right corner
    assert a.update_state(110, 0) == 110
    assert a.update_state(110, 1) == 110
    # test normal movement in the center of the grid
    assert a.update_state(53, 0) == 52
    assert a.update_state(53, 1) == 63
    assert a.update_state(45, 2) == 46
    assert a.update_state(64, 3) == 54
    # test wall collision
    assert a.update_state(24, 0) == 24
    assert a.update_state(26, 1) == 26
    assert a.update_state(34, 2) == 34
    assert a.update_state(46, 3) == 46
    # test termination on reaching the goal
    assert a.done is False
    a.update_state(91, 2)
    assert a.done is True
    # test termination on wholes
    a.done = False
    assert a.done is False
    a.update_state(72, 1)
    assert a.done is True


def test_epsilon_greedy_selection():
    """
    Test greedy selection and random tie breaking
    """
    a = Agent()
    a.q_values[107] = [0, 1, 0, 1]
    assert [a.epsilon_greedy(107) for _ in range(10)] == [3, 3, 2, 3, 3, 3, 1, 3, 1, 3]


def test_planning_step():
    a = Dyna_Q_Agent(planning_steps=10)
    a.update_model(0, 2, 1, 1)
    a.update_model(2, 0, 1, 1)
    a.update_model(0, 3, 0, 1)
    a.update_model(1, 1, -1, 1)

    expected_model = {
        0: {2: (1, 1), 3: (0, 1)},
        1: {1: (-1, 1)},
        2: {0: (1, 1)},
    }

    assert a.model == expected_model

    a.planning_step()
    expected_q_values = [np.array([0., 0., 0.1271, 0.2], dtype=np.float32),
                         np.array([0., 0.3439, 0., 0.], dtype=np.float32),
                         np.array([0.3152, 0., 0., 0.], dtype=np.float32)]
    assert np.all(np.isclose(expected_q_values, list(a.q_values.values())[:3]))


def test_agent_start_step_end(planning_steps=4,
                              epsilon=0.1,
                              gamma=1,
                              step_size=0.1):
    a = Dyna_Q_Agent(planning_steps=4, epsilon=0)

    # ----------------
    # test agent start
    # ----------------
    action = a.agent_start(a.start_position)
    assert action == 3
    assert a.position == 6
    assert a.model == {}
    for action_values in list(a.q_values.values()):
        assert np.all(action_values == 0)

    # ----------------
    # test agent step
    # ----------------
    action = a.step(a.position, a.env.get_reward(a.state_to_coord(a.position)))
    assert action == 1
    assert a.position == 16
    action = a.step(a.position, a.env.get_reward(a.state_to_coord(a.position)))
    assert action == 0
    assert a.position == 15
    action = a.step(a.position, a.env.get_reward(a.state_to_coord(a.position)))
    assert action == 2
    assert a.position == 16

    expected_model = {16: {3: (6, 0.0), 0: (15, 0.0)},
                      6: {1: (16, 0.0)}}
    assert a.model == expected_model

    for action_values in list(a.q_values.values()):
        assert np.all(action_values == 0)

    # ----------------
    # test agent end
    # ----------------
    # test the final update with a reward
    a.update_state(102, 3)
    a.agent_end()

    expected_q_values = np.array([0., 0., 0.271, 0.], dtype=np.float32)
    assert np.all(a.q_values.get(15) == expected_q_values)


def test_portal():
    a = Agent()
    assert a.position == 16
    a.update_state(107, 0)
    assert a.position == 110
    a.update_state(96, 1)
    assert a.position == 110
    a.update_state(105, 2)
    assert a.position == 110
    a.update_state(116, 3)
    assert a.position == 110
