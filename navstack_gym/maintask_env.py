import gym
import numpy as np
from . import base
from . import FOUND_IMMEDIATE_REWARD, DEFAULT_REWARD

class InvisibleTreasureHunt(base.InvisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.treasure_memo = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.treasure_memo = 0

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        # if np.sum(np.logical_and(self.unfound_chest)) - self.treasure_stock > 0:
        if self.treasure_memo != self.treasure_stock:
            reward += FOUND_IMMEDIATE_REWARD * (self.treasure_stock - self.treasure_memo)
            self.treasure_memo = self.treasure_stock

        return reward

class VisibleTreasureHunt(base.VisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.treasure_memo = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.treasure_memo = 0

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        # if np.sum(np.logical_and(self.unfound_chest)) - self.treasure_stock > 0:
        if self.treasure_memo != self.treasure_stock:
            reward += FOUND_IMMEDIATE_REWARD * (self.treasure_stock - self.treasure_memo)
            self.treasure_memo = self.treasure_stock

        return reward