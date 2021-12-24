from nav_sim_modules import ENV_SIZE, RESOLUTION, MAP_OBS_VAL, MAP_PASS_VAL, MAP_UNK_VAL, SPAWN_EXTENSION
import numpy as np

## registration OpenAI Gym
from gym.envs.registration import register

register(
    id='TreasureChestRoom-v0',
    entry_point='navstack_gym.base:TreasureChestEnv',
)
register(
    id='TreasureChestRoom-v1',
    entry_point='navstack_gym.base:VisibleTreasureChestEnv',
)
register(
    id='TreasureSearchRoom-v0',
    entry_point='navstack_gym.treasure_search_env:TreasureSearchEnv',
)
register(
    id='TreasureSearchRoom-v1',
    entry_point='navstack_gym.treasure_search_env:VisibleTreasureSearchEnv',
)

## General param
MAP_SIZE = ENV_SIZE
MAP_RESOLUTION = RESOLUTION

## HeuristicNavigation param
PATH_EXPLORATION_COUNT = 4000
PATH_PLANNING_COUNT = 10
PATH_TURNABLE = np.pi/8
ALLOWABLE_GOAL_ERROR_NORM = 0.5
AVOIDANCE_SIZE = 3
MOVE_LIMIT = -1

## Gym param
MOVABLE_DISCOUNT = 5 # movable = half_size / movable_discount
FOUND_THRESHOLD = 0.75
## randoor param
