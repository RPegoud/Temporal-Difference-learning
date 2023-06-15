from package.env import Env


def test_env_value_map_goal_positive_reward() -> None:
    env = Env()
    assert env.get_reward(env.coordinates.get('G')[0], reverse=False) == 1


def test_env_value_map_null_reward():
    env = Env()
    rows, cols = env.reward_map.shape
    for row in range(rows):
        for col in range(cols):
            if (row, col) != env.coordinates.get('G')[0]:
                assert env.get_reward((row, col), reverse=False) == 0
