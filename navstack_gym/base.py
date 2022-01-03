import gym
import numpy as np
import matplotlib.pyplot as plt

from descartes import PolygonPatch
from shapely.ops import unary_union
from collections import deque

from gym import spaces
from gym.utils import seeding
from nav_sim_modules import MAP_OBS_VAL, MAP_PASS_VAL, MAP_UNK_VAL, PASSABLE_COLOR

from nav_sim_modules.actioner import HeuristicAutonomousActioner
from nav_sim_modules.scener import ChestSearchRoomScener

from .utils import make_subjective_image, polar_to_cartesian_2d, relative_to_origin

from . import ALLOWABLE_GOAL_ERROR_NORM, AVOIDANCE_SIZE, MAP_SIZE, MAP_RESOLUTION, MOVABLE_DISCOUNT, MOVE_LIMIT, PATH_EXPLORATION_COUNT, PATH_PLANNING_COUNT, PATH_TURNABLE, SPAWN_EXTENSION , FOUND_THRESHOLD

class InvisibleTreasureChestRoom(gym.Env):

    # MAP_SIZE: 0.1
    # MAP_RESOLUTION: 256
    # MOVABLE_RANGE: HALF/5

    def __init__(self, 
            max_episode_steps: int=500, 
            seed=None,
            map_size: int=MAP_SIZE,
            map_resolition: float=MAP_RESOLUTION,
            spawn_extension: float=SPAWN_EXTENSION,
            path_exploration_count: int=PATH_EXPLORATION_COUNT,
            path_planning_count: int=PATH_PLANNING_COUNT,
            path_turnable: float=PATH_TURNABLE,
            allowable_goal_error_norm: float=ALLOWABLE_GOAL_ERROR_NORM,
            avoidance_size: int=AVOIDANCE_SIZE,
            move_limit: int=MOVE_LIMIT,
            found_threshold: float=FOUND_THRESHOLD, 
            passable_color: int=PASSABLE_COLOR,
            map_obs_val: int=MAP_OBS_VAL,
            map_pass_val: int=MAP_PASS_VAL,
            map_unk_val: int=MAP_UNK_VAL
    ):
        self.max_episode_steps = max_episode_steps
        self.map_size = map_size
        self.map_resolition = map_resolition
        self.found_threshold = found_threshold

        self.passable_color = passable_color
        self.map_obs_val = map_obs_val
        self.map_pass_val = map_pass_val
        self.map_unk_val = map_unk_val

        self.elapsed_step = 0

        self.agent_initial_position = (0,0,0) # (x,y,yaw)
        self.agent_current_position = (0,0,0)
        self.obstacle_positions = []
        self.key_positions = []
        self.chest_positions = []
        self.actioner = HeuristicAutonomousActioner(path_exploration_count, path_planning_count, path_turnable, allowable_goal_error_norm, avoidance_size, move_limit, map_resolition, passable_color, map_obs_val, map_pass_val, map_unk_val)
        self.scener = ChestSearchRoomScener(spawn_extension, map_size, map_resolition, passable_color, map_obs_val, map_pass_val)

        self.observation_space = spaces.Box(
            low=min([self.map_obs_val, self.map_pass_val, self.map_unk_val]),
            high=max([self.map_obs_val, self.map_pass_val, self.map_unk_val]),
            shape=(map_size,map_size,),
            dtype=np.int64
        )

        self.env_full_size = self.map_size * self.map_resolition
        self.env_half_size = self.env_full_size / 2
        self.moveble_range = self.env_half_size / MOVABLE_DISCOUNT
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.action_range = np.array([self.moveble_range, np.pi/2, np.pi/2])

        self.indecies_key = []
        self.indecies_chest = []
        self.unfound_key = []
        self.unfound_chest = []

        self.key_stock = 0
        self.treasure_stock = 0
        self.treasure_get_flag = False

        self.seed(seed)

    def reset(self, 
            is_generate_pose: bool=True,
            is_generate_room: bool=True,
            scene_obstacle_count: int=10,
            scene_obstacle_size: float=0.7,
            scene_target_size: float=0.2,
            scene_key_size: float=0.2,
            scene_obstacle_zone_thresh: float=1.5,
            scene_distance_key_placing: float=0.7,
            scene_range_key_placing: float=0.3, 
            scene_room_length_max: float=9,
            scene_room_wall_thickness: float=0.05, 
            scene_wall_threshold: float=0.1
    ):
        
        fl = self.scener.room_config is None

        if fl or is_generate_room:
            self._generate(scene_obstacle_count, scene_obstacle_size, scene_target_size, scene_key_size, scene_obstacle_zone_thresh, scene_distance_key_placing, scene_range_key_placing, scene_room_length_max, scene_room_wall_thickness, scene_wall_threshold)

            self.key_positions = self.scener.components_info[self.scener.tag_key]
            self.chest_positions = self.scener.components_info[self.scener.tag_chest]
            self.obstacle_positions = self.scener.components_info[self.scener.tag_obstacle]

        if fl or is_generate_pose:
            self.agent_initial_position = self.scener.spawn()
            self.agent_current_position = self.agent_initial_position

        self.unfound_key = np.ones([self.scener.room_config.key_count], dtype=np.bool8)
        self.unfound_chest = np.ones([self.scener.room_config.target_count], dtype=np.bool8)
        self.indecies_key = np.arange(len(self.unfound_key))
        self.indecies_chest = np.arange(len(self.unfound_chest))

        self.key_stock = 0
        self.treasure_stock = 0
        self.treasure_get_flag = False

        self.actioner.initialize(self.scener.env_pixel, self.agent_initial_position)
        self.elapsed_step = 0

        self._additional_reset()

        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action)
        self.elapsed_step += 1
        goal = self._convert_action_to_goal(action)
        self.actioner.do_action(goal)
        self.agent_current_position = self.actioner.pose
        self._check_found()
        reward = self._reward()
        done = self._is_done()
        observation = self._get_observation()
        info = []
        return observation, reward, done, info

    def step_with_debug(self, action, output_name=''):
        assert self.action_space.contains(action)
        self.elapsed_step += 1
        goal = self._convert_action_to_goal(action)
        print(f'now: {self.agent_current_position}, \nto: {goal}')
        self.actioner.do_action_visualize(goal, f'{output_name}_step_{str(self.elapsed_step).zfill(3)}')
        self.agent_current_position = self.actioner.pose
        self._check_found()
        reward = self._reward()
        done = self._is_done()
        observation = self._get_observation()
        info = []
        return observation, reward, done, info

    def _check_near(self) -> tuple:
        norms_chest = np.linalg.norm(self.chest_positions[:,:2] - self.agent_current_position[:2], axis=1)
        norms_key = np.linalg.norm(self.key_positions[:,:2] - self.agent_current_position[:2], axis=1)
        found_chest = self.indecies_chest[np.logical_and(self.unfound_chest, norms_chest <= self.found_threshold)]
        found_key = self.indecies_key[np.logical_and(self.unfound_key, norms_key <= self.found_threshold)]
        return found_chest, found_key

    def _check_found(self) -> None:
        found_chest, found_key = self._check_near()

        if len(found_key) > 0:
            self.unfound_key[found_key[0]] = False
            self.key_stock += 1
        elif (len(found_chest) > 0) and (self.key_stock > 0):
            self.key_stock -= 1
            self.treasure_get_flag = True
            self.unfound_chest[found_chest[0]] = False

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            env_size = self.map_resolition*self.map_size/2
            room = self.scener.room_config
            pose = self.agent_current_position

            fig = plt.figure(0, figsize=(9,9), dpi=60)
            ax = fig.add_subplot(111)
            plt.xlim(-env_size, env_size)
            plt.ylim(-env_size, env_size)

            wall = unary_union(room.get_polygons(room.tag_wall))
            obstacles = unary_union(room.get_polygons(room.tag_obstacle))
            chests = unary_union(room.get_polygons(room.tag_target))
            keys = unary_union(room.get_polygons(room.tag_key))
            obs_zones = unary_union(room.obstacle_hulls)
            key_zones = unary_union(room.key_placing_area)

            ax.add_patch(PolygonPatch(wall, fc='black', alpha=0.5, zorder=1))
            ax.add_patch(PolygonPatch(obstacles, fc='black', alpha=0.5, zorder=3, label="obstacle"))
            ax.add_patch(PolygonPatch(chests, fc='cyan', alpha=0.5, zorder=4, label='chest'))
            ax.add_patch(PolygonPatch(keys, fc='yellow', alpha=1, zorder=6, label='key'))
            ax.add_patch(PolygonPatch(key_zones, fc='yellow', alpha=0.1, zorder=5, label='key zone'))
            ax.add_patch(PolygonPatch(obs_zones, fc='black', alpha=0.1, zorder=4, label='obs zone'))

            found_chest = self.chest_positions[np.logical_not(self.unfound_chest)]
            found_key = self.key_positions[np.logical_not(self.unfound_key)]
            if len(found_chest) > 0:
                ax.scatter(found_chest[:,0], found_chest[:,1], color='red', s=3, zorder=7)
            if len(found_key) > 0:
                ax.scatter(found_key[:,0], found_key[:,1], color='red', s=3, zorder=7)

            r = 1
            angle_x = pose[0] + np.cos(pose[2])*r
            angle_y = pose[1] + np.sin(pose[2])*r
            ax.plot([pose[0],angle_x], [pose[1],angle_y], label='agent angle', zorder=4)

            ax.scatter(*pose[:2], s=50, color='red', label='agent position', zorder=4)

            map_img = np.copy(self.actioner.occupancy_map.T[::-1,:])
            map_img[map_img==self.map_unk_val] = 25
            map_img[map_img==self.map_pass_val] = 50
            map_img[map_img==self.map_obs_val] = 0

            ax.imshow(map_img, cmap='gray', alpha=0.8, extent=(-env_size,env_size,-env_size,env_size), zorder=3)

            ax.legend()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (w,h,4)
            buf = np.roll(buf, 3, axis=2)
            return buf
        
    def _generate(self, scene_obstacle_count, scene_obstacle_size, scene_target_size, scene_key_size, scene_obstacle_zone_thresh, scene_distance_key_placing, scene_range_key_placing, scene_room_length_max, scene_room_wall_thickness, scene_wall_threshold):

        self.scener.generate_scene(
            scene_obstacle_count,
            scene_obstacle_size,
            scene_target_size,
            scene_key_size,
            scene_obstacle_zone_thresh,
            scene_distance_key_placing,
            scene_range_key_placing,
            scene_room_length_max,
            scene_room_wall_thickness,
            scene_wall_threshold
        )

    def _additional_reset(self):
        pass

    def _get_observation(self) -> np.ndarray:
        return make_subjective_image(
            self.actioner.occupancy_map,
            self.actioner.pose[0] / self.map_resolition,
            self.actioner.pose[1] / self.map_resolition,
            self.actioner.pose[2],
            order=0,
            cval=MAP_UNK_VAL
        )
    
    def _convert_action_to_goal(self, relative_polar) -> tuple:
        act = self.action_range * relative_polar
        return relative_to_origin(*[*polar_to_cartesian_2d(*act[:2]), act[2]], *self.agent_current_position)
    
    def seed(self, seed=None):
        # np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_done(self) -> bool:
        return False

    def _reward(self) -> float:
        return 0

class VisibleTreasureChestRoom(InvisibleTreasureChestRoom):

    def _generate(self, scene_obstacle_count, scene_obstacle_size, scene_target_size, scene_key_size, scene_obstacle_zone_thresh, scene_distance_key_placing, scene_range_key_placing, scene_room_length_max, scene_room_wall_thickness, scene_wall_threshold):

        self.scener.generate_scene(
            scene_obstacle_count,
            scene_obstacle_size,
            scene_target_size,
            scene_key_size,
            scene_obstacle_zone_thresh,
            scene_distance_key_placing,
            scene_range_key_placing,
            scene_room_length_max,
            scene_room_wall_thickness,
            scene_wall_threshold, 
            True,
            True
        )

    def _additional_reset(self):
        self.scener.tweak_chest_collision_all(True)
        self.scener.tweak_key_collision_all(True)

    def _check_found(self) -> None:
        found_chest, found_key = self._check_near()

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