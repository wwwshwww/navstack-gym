import gym
import numpy as np
from . import base

from . import ALLOWABLE_GOAL_ERROR_NORM, AVOIDANCE_SIZE, MAP_SIZE, MAP_RESOLUTION, MOVABLE_DISCOUNT, MOVE_LIMIT, PATH_EXPLORATION_COUNT, PATH_PLANNING_COUNT, PATH_TURNABLE, SPAWN_EXTENSION , FOUND_THRESHOLD

class TreasureSearchEnv(base.TreasureChestEnv):

    def __init__(self, 
                max_episode_steps: int = 500, 
                seed=None, 
                map_size: int = MAP_SIZE, 
                map_resolition: float = MAP_RESOLUTION, 
                spawn_extension: float = SPAWN_EXTENSION, 
                path_exploration_count: int = PATH_EXPLORATION_COUNT, 
                path_planning_count: int = PATH_PLANNING_COUNT, 
                path_turnable: float = PATH_TURNABLE, 
                allowable_goal_error_norm: float = ALLOWABLE_GOAL_ERROR_NORM, 
                avoidance_size: int = AVOIDANCE_SIZE, 
                move_limit: int = MOVE_LIMIT, 
                found_threshold: float = FOUND_THRESHOLD):

        super().__init__(max_episode_steps=max_episode_steps, seed=seed, map_size=map_size, map_resolition=map_resolition, spawn_extension=spawn_extension, path_exploration_count=path_exploration_count, path_planning_count=path_planning_count, path_turnable=path_turnable, allowable_goal_error_norm=allowable_goal_error_norm, avoidance_size=avoidance_size, move_limit=move_limit, found_threshold=found_threshold)

        self.key_stock = 0
        self.treasure_stock = 0
        self.treasure_get_flag = False

    def reset(self, is_generate_pose: bool = True, is_generate_room: bool = True, scene_obstacle_count: int = 10, scene_obstacle_size: float = 0.7, scene_target_size: float = 0.2, scene_key_size: float = 0.2, scene_obstacle_zone_thresh: float = 1.5, scene_distance_key_placing: float = 0.7, scene_range_key_placing: float = 0.3, scene_room_length_max: float = 9, scene_room_wall_thickness: float = 0.05, scene_wall_threshold: float = 0.1):

        obs = super().reset(is_generate_pose=is_generate_pose, is_generate_room=is_generate_room, scene_obstacle_count=scene_obstacle_count, scene_obstacle_size=scene_obstacle_size, scene_target_size=scene_target_size, scene_key_size=scene_key_size, scene_obstacle_zone_thresh=scene_obstacle_zone_thresh, scene_distance_key_placing=scene_distance_key_placing, scene_range_key_placing=scene_range_key_placing, scene_room_length_max=scene_room_length_max, scene_room_wall_thickness=scene_room_wall_thickness, scene_wall_threshold=scene_wall_threshold)

        self.key_stock = 0
        self.treasure_stock = 0
        self.treasure_get_flag = False

        return obs

    def check_found(self) -> None:
        found_chest, found_key = self.check_near()

        if len(found_key) > 0:
            self.unfound_key[found_key[0]] = False
            self.key_stock += 1
        elif (len(found_chest) > 0) and (self.key_stock > 0):
            self.key_stock -= 1
            self.treasure_get_flag = True
            self.unfound_chest[found_chest[0]] = False

    def _reward(self) -> float:
        reward = 0
        # if np.sum(np.logical_and(self.unfound_chest)) - self.treasure_stock > 0:
        if self.treasure_get_flag:
            reward += 50
            self.treasure_stock += 1
            self.treasure_get_flag = False

        return reward

class VisibleTreasureSearchEnv(base.VisibleTreasureChestEnv):

    def __init__(self, 
                max_episode_steps: int = 500, 
                seed=None, 
                map_size: int = MAP_SIZE, 
                map_resolition: float = MAP_RESOLUTION, 
                spawn_extension: float = SPAWN_EXTENSION, 
                path_exploration_count: int = PATH_EXPLORATION_COUNT, 
                path_planning_count: int = PATH_PLANNING_COUNT, 
                path_turnable: float = PATH_TURNABLE, 
                allowable_goal_error_norm: float = ALLOWABLE_GOAL_ERROR_NORM, 
                avoidance_size: int = AVOIDANCE_SIZE, 
                move_limit: int = MOVE_LIMIT, 
                found_threshold: float = FOUND_THRESHOLD):

        super().__init__(max_episode_steps=max_episode_steps, seed=seed, map_size=map_size, map_resolition=map_resolition, spawn_extension=spawn_extension, path_exploration_count=path_exploration_count, path_planning_count=path_planning_count, path_turnable=path_turnable, allowable_goal_error_norm=allowable_goal_error_norm, avoidance_size=avoidance_size, move_limit=move_limit, found_threshold=found_threshold)

        self.key_stock = 0
        self.treasure_stock = 0
        self.treasure_get_flag = False

    def reset(self, is_generate_pose: bool = True, is_generate_room: bool = True, scene_obstacle_count: int = 10, scene_obstacle_size: float = 0.7, scene_target_size: float = 0.2, scene_key_size: float = 0.2, scene_obstacle_zone_thresh: float = 1.5, scene_distance_key_placing: float = 0.7, scene_range_key_placing: float = 0.3, scene_room_length_max: float = 9, scene_room_wall_thickness: float = 0.05, scene_wall_threshold: float = 0.1):

        obs = super().reset(is_generate_pose=is_generate_pose, is_generate_room=is_generate_room, scene_obstacle_count=scene_obstacle_count, scene_obstacle_size=scene_obstacle_size, scene_target_size=scene_target_size, scene_key_size=scene_key_size, scene_obstacle_zone_thresh=scene_obstacle_zone_thresh, scene_distance_key_placing=scene_distance_key_placing, scene_range_key_placing=scene_range_key_placing, scene_room_length_max=scene_room_length_max, scene_room_wall_thickness=scene_room_wall_thickness, scene_wall_threshold=scene_wall_threshold)

        self.key_stock = 0
        self.treasure_stock = 0
        self.treasure_get_flag = False

        return obs

    def check_found(self) -> None:
        found_chest, found_key = self.check_near()

        if len(found_key) > 0:
            self.unfound_key[found_key[0]] = False
            self.scener.tweak_key_collision(found_key[0], False)
            self.key_stock += 1
            
            new_env_pixel = self.scener.pixelize()
            self.actioner.register_env_pixel(new_env_pixel)
            self.actioner.navs.mapper.scan()
        elif (len(found_chest) > 0) and (self.key_stock > 0):
            self.key_stock -= 1
            self.treasure_get_flag = True
            self.unfound_chest[found_chest[0]] = False
            self.scener.tweak_chest_collision(found_chest[0], False)
            
            new_env_pixel = self.scener.pixelize()
            self.actioner.register_env_pixel(new_env_pixel)
            self.actioner.navs.mapper.scan()

    def _reward(self) -> float:
        reward = 0
        # if np.sum(np.logical_and(self.unfound_chest)) - self.treasure_stock > 0:
        if self.treasure_get_flag:
            reward += 50
            self.treasure_stock += 1
            self.treasure_get_flag = False

        return reward