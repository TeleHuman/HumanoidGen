import pdb, pickle, os
import numpy as np
import open3d as o3d
from copy import deepcopy
import zarr, shutil
import argparse
import yaml
from humanoidgen import ROOT_PATH

def main():
    
    with open(ROOT_PATH/"config/config_pkl2zarr.yml", "r") as file:
        zarr_config = yaml.safe_load(file)
        dataset_names=zarr_config["dataset_names"]
        episode_num=zarr_config["episode_num"]
        dp=zarr_config["dp"]
        dp3=zarr_config["dp3"]
        dp_downsample= zarr_config["dp_downsample"]
        dp3_downsample= zarr_config["dp3_downsample"]

    if dp == dp3:
        raise ValueError("dp and dp3 cannot be both True or both False. Please set only one to True.")
    
    for dataset_name in dataset_names:
        print(f'Processing datasets: {dataset_name}')
        ep_begin, ep_end = episode_num[0], episode_num[1]
        ep_p = ep_begin
        load_dir = str(ROOT_PATH/f'datasets/{dataset_name}')
        total_count = 0

        zarr_data_dp3 = None
        zarr_meta_dp3 = None

        zarr_data_dp = None
        zarr_meta_dp = None

        save_dataset_name = dataset_name.replace("_pkl", "")
        if dp3:
            save_dir_dp3 = str(ROOT_PATH/f'policy/3D-Diffusion-Policy/data/{save_dataset_name}_dp3.zarr')
            if os.path.exists(save_dir_dp3):
                shutil.rmtree(save_dir_dp3)
            zarr_root_dp3 = zarr.group(save_dir_dp3)
            zarr_data_dp3 = zarr_root_dp3.create_group('data')
            zarr_meta_dp3 = zarr_root_dp3.create_group('meta')

        if dp:
            save_dir_dp = str(ROOT_PATH/f'policy/Diffusion-Policy/data/{save_dataset_name}_dp.zarr')
            if os.path.exists(save_dir_dp):
                shutil.rmtree(save_dir_dp)
            zarr_root_dp = zarr.group(save_dir_dp)
            zarr_data_dp = zarr_root_dp.create_group('data')
            zarr_meta_dp = zarr_root_dp.create_group('meta')

        point_cloud_arrays, episode_ends_arrays, state_arrays, joint_action_arrays, head_camera_arrays = [], [], [], [], []
        
        while  ep_p < ep_end:

            if not os.path.isdir(load_dir+f'/episode{ep_p}'):
                ep_p += 1
                continue
            
            print(f'processing episode: {ep_p+1} / {ep_end}', end='\r')
            file_num = 0
            point_cloud_sub_arrays = []
            state_sub_arrays = []
            joint_action_sub_arrays = []
            head_camera_sub_arrays = []

            while os.path.exists(load_dir+f'/episode{ep_p}'+f'/{file_num}.pkl'):

                with open(load_dir+f'/episode{ep_p}'+f'/{file_num}.pkl', 'rb') as file:
                    data = pickle.load(file)

                joint_action = data['joint_action']
                joint_state = data['joint_state']

                state_sub_arrays.append(joint_state)
                joint_action_sub_arrays.append(joint_action)
                
                if dp3:
                    pcd = data['pointcloud']
                    point_cloud_sub_arrays.append(pcd)
                    file_num += dp3_downsample
                
                if dp:
                    head_img = data['observation']['head_camera']['rgb']
                    head_camera_sub_arrays.append(head_img)
                    file_num += dp_downsample
                total_count += 1
            
            ep_p += 1
            total_count -= 1

            joint_action_sub_arrays = np.array(joint_action_sub_arrays)
            state_sub_arrays = np.array(state_sub_arrays)

            joint_action_sub_arrays = joint_action_sub_arrays[1:]
            # joint_action_sub_arrays[:,0:7] = state_sub_arrays[1:,0:7] # left arm
            # joint_action_sub_arrays[:,13:20] = state_sub_arrays[1:,13:20] # right arm

            state_sub_arrays = state_sub_arrays[:-1]


            episode_ends_arrays.append(deepcopy(total_count))
            state_arrays.extend(state_sub_arrays)
            joint_action_arrays.extend(joint_action_sub_arrays)
            
            if dp3:
                point_cloud_sub_arrays = point_cloud_sub_arrays[:-1]
                point_cloud_arrays.extend(point_cloud_sub_arrays)

            if dp:
                head_camera_sub_arrays = head_camera_sub_arrays[:-1]
                head_camera_arrays.extend(head_camera_sub_arrays)

        episode_ends_arrays = np.array(episode_ends_arrays)
        joint_action_arrays =np.array(joint_action_arrays)
        state_arrays = np.array(state_arrays)
        if dp3:
            point_cloud_arrays = np.array(point_cloud_arrays)
        if dp:
            head_camera_arrays = np.array(head_camera_arrays)
            head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

        if dp3:
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            state_chunk_size = (100, state_arrays.shape[1])
            joint_chunk_size = (100, joint_action_arrays.shape[1])
            point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])

            zarr_data_dp3.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data_dp3.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data_dp3.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_meta_dp3.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
        if dp:
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            state_chunk_size = (100, state_arrays.shape[1])
            joint_chunk_size = (100, joint_action_arrays.shape[1])
            head_camera_chunk_size = (100, head_camera_arrays.shape[1], head_camera_arrays.shape[2], head_camera_arrays.shape[3])

            zarr_data_dp.create_dataset('head_camera', data=head_camera_arrays, chunks=head_camera_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
            zarr_data_dp.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data_dp.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_meta_dp.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()