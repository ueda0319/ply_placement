#import open3d
import numpy as np
import glob
import random
from typing import List
import os
import pickle

from label_mapping import load_label_mapping

class_labels = load_label_mapping()

def generate_floor(point_count: int = 16384, floor_l: float = 8.0, floor_w: float = 8.0, floor_h: float = 0.03):
    floor_points = np.random.rand(point_count, 6)
    floor_points[:, 0] *= floor_l
    floor_points[:, 1] *= floor_w
    floor_points[:, 2] *= floor_h
    floor_points[:, 3:] = floor_points[:, 3:] * 0.1 + 0.5
    return floor_points

def generate_random_object():
    label = random.choice(list(class_labels.keys()))
    category_id = class_labels[label]
    filenames = glob.glob(f'data/{label}/{label}_????.ply')
    filename = random.choice(filenames)
    scale = np.random.rand() * 0.6 + 1.0
    trans = np.random.rand(1, 3)*6+1
    rot_z = np.random.rand() * 2 * 3.1415
    rot_mat = np.array([[np.cos(rot_z), -np.sin(rot_z), 0.0],[np.sin(rot_z), np.cos(rot_z), 0.0],[0.0,0.0,1.0]])

    obj = open3d.io.read_point_cloud(filename, format="ply")
    obj_points = np.asarray(obj.points)
    obj_colors = np.asarray(obj.colors) #np.repeat(np.random.rand(1, 3), obj_points.shape[0], 0)
    obj_points *= scale
    obj_points = np.matmul(obj_points, rot_mat)
    trans[0, 2] = - np.min(obj_points[:, 2])
    obj_points += trans
    return np.concatenate([obj_points, obj_colors], 1), category_id

def get_bb(obj_np: np.ndarray):
    return np.array([obj_np[:,0].min(), obj_np[:,1].min(), obj_np[:,0].max(), obj_np[:,1].max()])

def is_hit_bb(bb: np.ndarray, bb_registered: List[np.ndarray]):
    for bbr in bb_registered:
        if bbr[0] < bb[2] and bb[0] < bbr[2] and bbr[1] < bb[3] and bb[1] < bbr[3]:
            return True
    return False

def farthest_point_sample(point, npoint=16384):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
    

def numpy2o3d(obj_np):
    obj = open3d.geometry.PointCloud()
    obj.points = open3d.utility.Vector3dVector(obj_np[:,:3])
    obj.colors = open3d.utility.Vector3dVector(obj_np[:,3:])
    return obj

def generate_info_dict(save_dir: str = "output", scene_name: str = "scene0000"):
    info = {
        "point_cloud": {
            {'num_features': 6, 'lidar_idx': sample_id}
        },
        "pts_path": f"{save_dir}/points/{scene_name}.bin",
        "super_pts_path": f"{save_dir}/super_points/{scene_name}.bin",
        "pts_instance_mask_path": f"{save_dir}/instance_mask/{scene_name}.bin",
        "pts_semantic_mask_path": f"{save_dir}/semantic_mask/{scene_name}.bin",
    }

def main():
    save_dir: str = "output"
    # Create save directories
    os.makedirs(f"{save_dir}/points", exist_ok=True)
    os.makedirs(f"{save_dir}/semantic_mask", exist_ok=True)
    os.makedirs(f"{save_dir}/instance_mask", exist_ok=True)
    os.makedirs(f"{save_dir}/seg_info", exist_ok=True)

    infos = []

    for scene_id in range(10):
        scene_name: str = "scene{:04}".format(scene_id)
        bb_list = []
        points = generate_floor()
        semantic_mask = np.zeros((points.shape[0],), dtype=np.int64)
        instance_mask = np.zeros((points.shape[0],), dtype=np.int64)
        o3dobj_list = [numpy2o3d(generate_floor())]
        for i in range(10):
            obj, category_id = generate_random_object()
            obj_bb = get_bb(obj)
            if len(bb_list)==0 or not is_hit_bb(obj_bb, bb_list):
                points = np.concatenate([points, obj], 0)
                semantic_mask = np.concatenate([semantic_mask, category_id * np.ones((obj.shape[0],), dtype=np.int64)])
                instance_mask = np.concatenate([instance_mask, (i+1) * np.ones((obj.shape[0],), dtype=np.int64)])
                o3dobj_list.append(numpy2o3d(obj))
                bb_list.append(obj_bb)

        # Reduce point count when scene have too much points
        max_num_points = 65536
        if points.shape[0] > max_num_points:
            choices = np.random.choice(points.shape[0], max_num_point, replace=False)
            points = points[choices, :]
            semantic_mask = semantic_mask[choices]
            instance_mask = instance_mask[choices]

        points.tofile(f"{save_dir}/points/{scene_name}.bin")
        semantic_mask.tofile(f"{save_dir}/semantic_mask/{scene_name}.bin")
        instance_mask.tofile(f"{save_dir}/instance_mask/{scene_name}.bin")
        info = {
            "point_cloud": {
                'num_features': 6, 
                'lidar_idx': scene_name
            },
            "pts_path": f"{save_dir}/points/{scene_name}.bin",
            "super_pts_path": f"{save_dir}/super_points/{scene_name}.bin",
            "pts_instance_mask_path": f"{save_dir}/instance_mask/{scene_name}.bin",
            "pts_semantic_mask_path": f"{save_dir}/semantic_mask/{scene_name}.bin",
        }
        infos.append(info)
    with open(f"{save_dir}/scannet_infos_train.pkl", mode='wb') as fo:
        pickle.dump(infos, fo)


if __name__ == "__main__":
    main()