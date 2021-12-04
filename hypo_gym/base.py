import gym
from gym import spaces
import numpy as np
from nav_sim_modules.actioner import HeuristicAutonomousActioner
from nav_sim_modules.scener import ChestSearchRoomScener

from typing import Sequence

from . import MAP_SIZE, MAP_RESOLUTION, MOVABLE_RANGE

class ChestSearchEnv(gym.Env):

    # MAP_SIZE: 0.1
    # MAP_RESOLUTION: 256

    def __init__(self, max_episode_steps=500, **kwargs):
        self.agent_postion = (0,0,0) # (x,y,yaw)
        self.key_postions = []
        self.chest_potions = []
        self.scener = HeuristicAutonomousActioner(MAP_RESOLUTION)
        self.actioner = ChestSearchRoomScener(MAP_SIZE, MAP_RESOLUTION)

        self.map_pass_val = self.actioner.navs.map_pass_val
        self.map_obs_val = self.actioner.navs.map_obs_val
        self.map_unk_val = self.actioner.navs.map_unk_val

        self.observation_space = spaces.Box(
            low=min([self.map_obs_val, self.map_pass_val, self.map_unk_val]),
            high=max([self.map_obs_val, self.map_pass_val, self.map_unk_val]),
            shape=(MAP_SIZE,MAP_SIZE,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.action_range = np.array([MOVABLE_RANGE, np.pi/2, np.pi])
        self.seed()


    def reset(self, generating_params: Sequence, is_generate_pose: bool,is_generate_room: bool):
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

    
    def seed(self, seed=None):
        np.random.seed(seed)
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]

    def _is_done(self):
        pass

    def reward(self):
        pass