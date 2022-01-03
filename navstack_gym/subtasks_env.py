import gym
import numpy as np
from . import base
from . import FOUND_IMMEDIATE_REWARD, DEFAULT_REWARD, EXPLORE_MAGNIFICATION_REWARD

class InvisibleKeyHunt(base.InvisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_found_num = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.key_found_num = 0

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        if self.key_found_num < np.sum(self.unfound_key==False):
            self.key_found_num += 1
            reward += FOUND_IMMEDIATE_REWARD
        return reward

class VisibleKeyHunt(base.VisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_found_num = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.key_found_num = 0

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        if self.key_found_num < np.sum(self.unfound_key==False):
            self.key_found_num += 1
            reward += FOUND_IMMEDIATE_REWARD
        return reward


class InvisibleChestHunt(base.InvisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chest_found_num = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.chest_found_num = 0

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        if self.chest_found_num < np.sum(self.unfound_chest==False):
            self.chest_found_num += 1
            reward += FOUND_IMMEDIATE_REWARD
        return reward

class InvisibleChestHunt(base.InvisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chest_found_num = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.chest_found_num = 0

    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        if self.chest_found_num < np.sum(self.unfound_chest==False):
            self.chest_found_num += 1
            reward += FOUND_IMMEDIATE_REWARD
        return reward

class InvisibleMapExplore(base.InvisibleTreasureChestRoom):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_passable = 0
        self.residual_passable = 0
        self.total_passable = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.total_passable = np.sum(self.scener.get_current_env_pixel() == self.map_pass_val)
        self.current_passable = np.sum(self.actioner.occupancy_map == self.map_pass_val)
        self.residual_passable = self.total_passable - self.current_passable
    
    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        next_passable = np.sum(self.actioner.occupancy_map == self.map_pass_val)
        reward += (next_passable - self.current_passable) / self.residual_passable * EXPLORE_MAGNIFICATION_REWARD
        self.current_passable = next_passable
        return reward

class VisibleMapExplore(base.VisibleTreasureChestRoom):
    """
    実質最強の報酬設計かもしれない。これだけで宝探しをクリアできてしまう可能性がある。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_passable = 0
        self.residual_passable = 0
        self.total_passable = 0

    def _additional_reset(self):
        super()._additional_reset()
        self.total_passable = np.sum(self.scener.get_current_env_pixel() == self.map_pass_val)
        self.current_passable = np.sum(self.actioner.occupancy_map == self.map_pass_val)
        self.residual_passable = self.total_passable - self.current_passable
    
    def _reward(self) -> float:
        reward = DEFAULT_REWARD
        next_passable = np.sum(self.actioner.occupancy_map == self.map_pass_val)
        reward += (next_passable - self.current_passable) / self.residual_passable * EXPLORE_MAGNIFICATION_REWARD
        self.current_passable = next_passable
        return reward