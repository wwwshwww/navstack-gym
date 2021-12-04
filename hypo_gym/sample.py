import gym
import numpy as np
from nav_sim_modules.actioner import HeuristicAutonomousActioner
from nav_sim_modules.scener import ChestSearchRoomScener

class TakarasagashiEnv(gym.Env):
    def __init__(self, max_episode_steps=500, **kwargs):
        self.agent_postion = [0,0,0] # (x,y,yaw)
        self.key_postions = []
        self.chest_potions = []
        self.scener = HeuristicAutonomousActioner()
        self.actioner = ChestSearchRoomScener()

    def reset(self, generating_params: list, is_generate_pose: bool,is_generate_room: bool) -> np.ndarray:
        pass

    def step(self, action):
        self.agen_position = self.actioner.goto(action)
        observation = self._subjective(self.actioner.navs.mapper.occupancy_map)
        done = self._is_done()
        reward = self.reward()
        info = []
        return observation, reward, done, info
        
    def _subjective(self, map):
        # affine transform with agent_position
        pass

    def _is_done(self):
        pass

    def reward(self):
        pass