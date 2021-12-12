from nav_sim_modules import ENV_SIZE, RESOLUTION, MAP_OBS_VAL, MAP_PASS_VAL, MAP_UNK_VAL, SPAWN_EXTENSION
import numpy as np

## registration OpenAI Gym
from gym.envs.registration import register

register(
    id='TreasureChestRoom-v0',
    entry_point='navstack_gym.base:TreasureChestEnv',
)

## General param
MAP_SIZE = ENV_SIZE
MAP_RESOLUTION = RESOLUTION

## HeuristicNavigation param
PATH_EXPLORATION_COUNT = 10000
PATH_PLANNING_COUNT = 10
PATH_TURNABLE = np.pi/8
ALLOWABLE_GOAL_ERROR_NORM = 0.5
AVOIDANCE_SIZE = 2
MOVE_LIMIT = -1

## Gym param
MOVABLE_DISCOUNT = 5 # half_size / <"this">

## randoor param
