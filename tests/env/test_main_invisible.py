import gym
import navstack_gym
import pytest

@pytest.fixture(scope='module')
def env(request):
    env = gym.make('InvisibleTreasureHunt-v0')
    yield env

@pytest.mark.commit 
def test_initialization(env):
    env.reset(is_generate_room=True, is_generate_pose=True)
    done = False
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)

    assert env.observation_space.contains(observation)