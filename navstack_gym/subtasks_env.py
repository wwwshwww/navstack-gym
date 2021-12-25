import gym
import numpy as np
from . import base
from . import FOUND_IMMEDIATE_REWARD

class InvisibleKeyHunt(base.InvisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_found_num = 0
        self.reward_flag = False

    def _additional_reset(self):
        super()._additional_reset()
        self.key_found_num = 0
        self.reward_flag = False

    def _check_found(self) -> None:
        _, found_key = self._check_near()
        if len(found_key) > 0:
            self.key_found_num += 1
            self.reward_flag = True
            self.unfound_key[found_key[0]] = False

    def _reward(self) -> float:
        reward = 0
        if self.reward_flag:
            reward += FOUND_IMMEDIATE_REWARD
            self.reward_flag = False
        return reward

class VisibleKeyHunt(base.VisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_found_num = 0
        self.reward_flag = False

    def _additional_reset(self):
        super()._additional_reset()
        self.key_found_num = 0
        self.reward_flag = False

    def _check_found(self) -> None:
        _, found_key = self._check_near()
        if len(found_key) > 0:
            self.unfound_key[found_key[0]] = False
            self.scener.tweak_key_collision(found_key[0], False)
            self.key_found_num += 1
            self.reward_flag = True
            
            new_env_pixel = self.scener.pixelize()
            self.actioner.register_env_pixel(new_env_pixel)
            self.actioner.navs.mapper.scan()

    def _reward(self) -> float:
        reward = 0
        if self.reward_flag:
            reward += FOUND_IMMEDIATE_REWARD
            self.reward_flag = False
        return reward



class InvisibleChestHunt(base.InvisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chest_found_num = 0
        self.reward_flag = False

    def _additional_reset(self):
        super()._additional_reset()
        self.chest_found_num = 0
        self.reward_flag = False

    def _check_found(self) -> None:
        found_chest, _ = self._check_near()
        if len(found_chest) > 0:
            self.chest_found_num += 1
            self.reward_flag = True
            self.unfound_chest[found_chest[0]] = False

    def _reward(self) -> float:
        reward = 0
        if self.reward_flag:
            reward += FOUND_IMMEDIATE_REWARD
            self.reward_flag = False
        return reward

class VisibleChestHunt(base.VisibleTreasureChestRoom):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chest_found_num = 0
        self.reward_flag = False

    def _additional_reset(self):
        super()._additional_reset()
        self.chest_found_num = 0
        self.reward_flag = False

    def _check_found(self) -> None:
        found_chest, _ = self._check_near()
        if len(found_chest) > 0:
            self.unfound_chest[found_chest[0]] = False
            self.scener.tweak_chest_collision(found_chest[0], False)
            self.chest_found_num += 1
            self.reward_flag = True
            
            new_env_pixel = self.scener.pixelize()
            self.actioner.register_env_pixel(new_env_pixel)
            self.actioner.navs.mapper.scan()

    def _reward(self) -> float:
        reward = 0
        if self.reward_flag:
            reward += FOUND_IMMEDIATE_REWARD
            self.reward_flag = False
        return reward
