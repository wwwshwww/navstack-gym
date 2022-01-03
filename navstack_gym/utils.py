import numpy as np
from scipy.ndimage import affine_transform
from PIL import Image

def normalize_angle_rad(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def transform_2d(target_x, target_y, target_yaw, origin_x, origin_y, origin_yaw):
    t_mat = np.identity(3)
    t_mat[:2,2] = [-origin_x, -origin_y]
    r_mat = np.array([
                [np.cos(-origin_yaw), -np.sin(-origin_yaw), 0],
                [np.sin(-origin_yaw), np.cos(-origin_yaw), 0],
                [0, 0, 1]
            ])
    af = np.dot(r_mat, t_mat)
    target_xy = np.array([target_x, target_y, 1])
    transed_xy = np.dot(af, target_xy)
    
    return transed_xy[0], transed_xy[1], normalize_angle_rad(target_yaw - origin_yaw)

def relative_to_origin(target_x, target_y, target_yaw, origin_x, origin_y, origin_yaw):
    origin_v = np.array([origin_x, origin_y])
    r_mat = np.array([
                [np.cos(origin_yaw), -np.sin(origin_yaw), 0],
                [np.sin(origin_yaw), np.cos(origin_yaw), 0],
                [0, 0, 1]
            ])
    target_v = np.array([target_x, target_y, 1])
    corrected_relative = np.dot(r_mat, target_v)
    
    ang = normalize_angle_rad(origin_yaw + target_yaw)
    
    return origin_x+corrected_relative[0], origin_y+corrected_relative[1], ang

def cartesian_to_polar_2d(x, y):
    return np.linalg.norm([x,y]), np.arctan2(y,x)

def polar_to_cartesian_2d(r, theta):
    return r*np.cos(theta), r*np.sin(theta)

def make_subjective_image(img, x, y, rad, order=1, cval=-1):
    result = np.copy(img)
    half_h, half_w = map(lambda x: x/2, img.shape)
    
    t1 = np.array([
        [1, 0, -half_h],
        [0, 1, -half_w],
        [0, 0, 1]
    ])
    
    r = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])
    
    t2 = np.array([
        [1, 0, half_h+x],
        [0, 1, half_w+y], 
        [0, 0, 1]
    ])
    affine = np.matmul(t2, np.matmul(r, t1))
    return affine_transform(result, affine, order=order, cval=cval)

def create_gif(frames: list, filename: str="output", mode='RGB', duration=60):
    frs = [Image.fromarray(f, mode=mode) for f in frames]
    frs[0].save(f'{filename}.gif', save_all=True, append_images=frs[1:], optimize=False, duration=duration, loop=0)