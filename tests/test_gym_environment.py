from gymnasium import spaces

from releso.gym_environment import GymEnvironment


def test_gym_environment_dummy():
    # This test is just a dummy test since this class does not really do
    # anything since its functionality is provided by the parser environment.
    env = GymEnvironment(spaces.Discrete(10), spaces.Box(0, 1, shape=(1, 1)))
    env.step(1)
    env.reset()
    env.render()
    env.close()
