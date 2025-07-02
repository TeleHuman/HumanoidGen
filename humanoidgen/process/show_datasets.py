import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import cv2
from humanoidgen import ROOT_PATH
import yaml
from enum import Enum
import time
import matplotlib.pyplot as plt

class RunType(Enum):
    SHOW_VIDEO = "show_video"
    SAVE_VIDEO = "save_video"
    SHOW_PC = "show_pc"
    SHOW_STATES = "show_states"

def main():
    # read the configuration file for displaying datasets
    with open(ROOT_PATH/"config/config_show_datasets.yml", "r") as file:
        show_config = yaml.safe_load(file)
        dataset_names = show_config["dataset_names"]
        episode_num = show_config["episode_num"]
        run_type_str = show_config["run_type"]
        fps = show_config["fps"]
        video_writer_type = show_config["video_writer_type"]
    run_type = RunType(run_type_str)
    
    for dataset_name in dataset_names:
        ep_begin, ep_end = episode_num[0], episode_num[1]
        ep_p = ep_begin
        load_dir = str(ROOT_PATH/f'datasets/{dataset_name}')
        if not os.path.exists(load_dir):
            print(f"Dataset {dataset_name} does not exist at {load_dir}, skipping...")
            continue
        save_video_dir = str(ROOT_PATH / 'videos'/'datasets'/f"{dataset_name}")  # Directory to save videos
        os.makedirs(save_video_dir, exist_ok=True)

        while ep_p < ep_end:
            if not os.path.isdir(load_dir + f'/episode{ep_p}'):
                print(f'episode {ep_p} does not exist, skipping...')
                ep_p += 1
                continue
            
            file_num = 0            

            video_writers = {}
            
            vis = None
            if run_type == RunType.SHOW_PC:
                import open3d as o3d
                # Initialize Open3D visualizer
                vis = o3d.visualization.Visualizer()
                vis.create_window()

            all_states = []
            while os.path.exists(load_dir + f'/episode{ep_p}' + f'/{file_num}.pkl'):
                print(f'processing episode: {ep_p+1} / {ep_end} , processing frame:{file_num}', end='\r')
                
                with open(load_dir + f'/episode{ep_p}' + f'/{file_num}.pkl', 'rb') as file:
                    data = pickle.load(file)
                
                if run_type == RunType.SHOW_VIDEO:
                    head_img_rgb = data['observation']['head_camera']['rgb']
                    head_img_bgr = head_img_rgb[..., ::-1]
                    cv2.imshow("Head Camera Image", head_img_bgr)
                    cv2.waitKey(1) # ms

                elif run_type == RunType.SAVE_VIDEO:
                    # for key, value in data['observation'].items():
                    for key, value in data['observation'].items():
                        if "rgb" in value:
                            img = value['rgb']
                            h,w,_ = img.shape # height, width, channels
                            if key not in video_writers:
                                os.makedirs(f"{save_video_dir}/episode_{ep_p}", exist_ok=True)
                                if video_writer_type == "opencv":
                                    video_writers[key] = cv2.VideoWriter(
                                        os.path.join(f"{save_video_dir}/episode_{ep_p}", f"{key}.mp4"),
                                        cv2.VideoWriter_fourcc(*'mp4v'),  # Codec for mp4
                                        fps,  # FPS
                                        (w, h)
                                    )
                                elif video_writer_type == "imageio":
                                    import imageio
                                    video_writers[key] = imageio.get_writer(
                                        os.path.join(f"{save_video_dir}/episode_{ep_p}", f"{key}.mp4"),
                                        fps=fps
                                    )
                                print(f"Creating video writer for {key} at {save_video_dir}/episode_{ep_p}/{key}.mp4")
                            if video_writer_type == "opencv":
                                video_writers[key].write(img[..., ::-1])
                            elif video_writer_type == "imageio":
                                # imageio uses RGB, so we can write directly
                                video_writers[key].append_data(img)  # Convert RGB to BGR for OpenCV
                
                elif run_type == RunType.SHOW_PC:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(data['pointcloud'][:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(data['pointcloud'][:, 3:])
                    vis.clear_geometries()
                    vis.add_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.1)

                elif run_type == RunType.SHOW_STATES:
                    joint_state = data['joint_state']
                    all_states.append(deepcopy(joint_state))

                file_num += 1
    
            if run_type == RunType.SAVE_VIDEO:
                for key, writer in video_writers.items():
                        if video_writer_type == "opencv":
                            writer.release()
                        elif video_writer_type == "imageio":
                            writer.close()

            elif run_type == RunType.SHOW_STATES:
                all_states = np.array(all_states)
                plt.figure(figsize=(10, 6))
                for i in range(all_states.shape[-1]):  
                    plt.plot(all_states[:,  i], label=f"State Dimension {i+1}")
                plt.xlabel("Step")
                plt.ylabel("State Value")
                plt.title("State Values Over Time")
                plt.legend()
                plt.grid()
                plt.show()

            ep_p += 1

if __name__ == '__main__':
    main()