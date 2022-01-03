from nav_sim_modules import ENV_SIZE, RESOLUTION, MAP_OBS_VAL, MAP_PASS_VAL, MAP_UNK_VAL, SPAWN_EXTENSION
import numpy as np

## registration OpenAI Gym
from gym.envs.registration import register

register(
    id='InvisibleTreasureChestRoom-v0',
    entry_point='navstack_gym.base:InvisibleTreasureChestRoom',
)
register(
    id='VisibleTreasureChestRoom-v0',
    entry_point='navstack_gym.base:VisibleTreasureChestRoom',
)
register(
    id='InvisibleTreasureHunt-v0',
    entry_point='navstack_gym.maintask_env:InvisibleTreasureHunt',
)
register(
    id='VisibleTreasureHunt-v0',
    entry_point='navstack_gym.maintask_env:VisibleTreasureHunt',
)
register(
    id='InvisibleKeyHunt-v0',
    entry_point='navstack_gym.subtasks_env:InvisibleKeyHunt',
)
register(
    id='VisibleKeyHunt-v0',
    entry_point='navstack_gym.subtasks_env:VisibleKeyHunt',
)
register(
    id='InvisibleChestHunt-v0',
    entry_point='navstack_gym.subtasks_env:InvisibleChestHunt',
)
register(
    id='VisibleChestHunt-v0',
    entry_point='navstack_gym.subtasks_env:VisibleChestHunt',
)
register(
    id='InvisibleMapExplore-v0',
    entry_point='navstack_gym.subtasks_env:InvisibleMapExplore',
)
register(
    id='VisibleMapExplore-v0',
    entry_point='navstack_gym.subtasks_env:VisibleMapExplore',
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
FOUND_IMMEDIATE_REWARD = 50
EXPLORE_MAGNIFICATION_REWARD = 200
DEFAULT_REWARD = -0.04
## randoor param
