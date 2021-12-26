import gym
import numpy as np
from . import base
from . import FOUND_IMMEDIATE_REWARD, DEFAULT_REWARD

class InvisibleTreasureHunt(base.InvisibleTreasureChestRoom):

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        # if np.sum(np.logical_and(self.unfound_chest)) - self.treasure_stock > 0:
        if self.treasure_get_flag:
            reward += FOUND_IMMEDIATE_REWARD
            self.treasure_stock += 1
            self.treasure_get_flag = False

        return reward

class VisibleTreasureHunt(base.VisibleTreasureChestRoom):

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        # if np.sum(np.logical_and(self.unfound_chest)) - self.treasure_stock > 0:
        if self.treasure_get_flag:
            reward += FOUND_IMMEDIATE_REWARD
            self.treasure_stock += 1
            self.treasure_get_flag = False

        return reward