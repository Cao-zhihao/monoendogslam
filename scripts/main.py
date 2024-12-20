import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.gradslam_datasets import (  
    load_dataset_config,
    EndoSLAMDataset,
    C3VDDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_means3D
from utils.eval_helpers import report_progress, eval_save
from utils.keyframe_selection import keyframe_selection_overlap, keyframe_selection_distance
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from utils.vis_utils import plot_video
from utils.time_helper import Timer

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

# 导入 monocular.py 中的类和函数
from monocular import CoarseInitializer, FrameData, show_img, save_img

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["endoslam_unity"]:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def initialize_params(init_pt_cld, num_frames, use_simplification=True):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.log(torch.ones_like(torch.sqrt(torch.ones(num_pts, 1).cuda())))  # 统一缩放
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45)  # 设置SH degree 3固定

    # 初始化单个高斯轨迹来建模相机姿态相对于第一帧
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables

def initialize_optimizer(params, lrs_dict):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k != 'feature_rest']
    if 'feature_rest' in params:
        param_groups.append({'params': [params['feature_rest']], 'name': 'feature_rest', 'lr': lrs['rgb_colors'] / 20.0})
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, use_simplification=True):
    # 加载第一帧的RGB图像
    color, _, _, _ = dataset[0]
    color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)

    # 初始化单目姿态估计器
    camMat = np.array([[723.7722, 0, 337.5],
                       [0, 723.7722, 270.0],
                       [0, 0, 1]], dtype=np.float32)  # 根据实际相机参数调整
    scale = 1.0  # 根据需要调整缩放比例
    initializer = CoarseInitializer(camMat, scale=scale, window_size=2, matches_ratio=0.7)

    # 初始化参数和变量
    params, variables = initialize_params(init_pt_cld=torch.empty(0), num_frames=num_frames, use_simplification=use_simplification)

    # 处理第一帧
    rst = initializer.forward(color.cpu().numpy())
    if rst is not None:
        pose, points = rst
        # 将点云和位姿添加到参数中
        params['means3D'] = torch.nn.Parameter(torch.tensor(points).cuda().float().requires_grad_(True))
        params['rgb_colors'] = torch.nn.Parameter(torch.zeros_like(params['means3D'][:, :3]).cuda().float())
        # 初始化相机位姿
        params['cam_trans'][..., 0] = torch.tensor(pose[:3, 3]).cuda().float()
        params['cam_unnorm_rots'][..., 0] = torch.tensor(pose[:3, :3]).cuda().float()

    # 初始化场景半径
    variables['scene_radius'] = torch.max(torch.tensor(points).cuda().float()) / scene_radius_depth_ratio

    return params, variables, camMat, initializer

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, 
             sil_thres, use_l1, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # 初始化损失字典
    losses = {}

    if tracking:
        # 获取当前帧的高斯，只有相机姿态有梯度
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:  # 捆绑调整
            # 获取当前帧的高斯，相机姿态和高斯都有梯度
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # 获取当前帧的高斯，只有高斯有梯度
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # 获取当前帧的高斯，只有高斯有梯度
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # 初始化渲染变量
    rendervar = transformed_params2rendervar(params, transformed_pts)

    # RGB 渲染
    rendervar['means2D'].retain_grad()
    im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # 仅颜色渲染的梯度用于稠密化

    # RGB 损失
    if tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True):
    # 添加新高斯的逻辑根据单目SLAM进行调整
    # 这里需要基于新的点云生成策略
    pass

def initialize_camera_pose(params, curr_time_idx, forward_prop):
    # 单目SLAM中相机姿态已经通过初始化器估计，无需额外处理
    return params

def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store

def rgbd_slam(config: dict):
    # timer = Timer()
    # timer.start()
    
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if 'distance_keyframe_selection' not in config:
        config['distance_keyframe_selection'] = False
    if config['distance_keyframe_selection']:
        print("Using CDF Keyframe Selection. Note that \'mapping window size\' is useless.")
        if 'distance_current_frame_prob' not in config:
            config['distance_current_frame_prob'] = 0.5
    if 'gaussian_simplification' not in config:
        config['gaussian_simplification'] = True # simplified in paper
    if not config['gaussian_simplification']:
        print("Using Full Gaussian Representation, which may cause unstable optimization if not fully optimized.")
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "train_or_test" not in dataset_config:
        dataset_config["train_or_test"] = 'all'
    if "preload" not in dataset_config:
        dataset_config["preload"] = False
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame

    dataset = get_dataset(
        config_dict=load_dataset_config(dataset_config["gradslam_data_cfg"]),
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        train_or_test=dataset_config["train_or_test"]
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # 初始化单目姿态估计器和参数
    camMat = np.array([[723.7722, 0, 337.5],
                       [0, 723.7722, 270.0],
                       [0, 0, 1]], dtype=np.float32)  # 根据实际相机参数调整
    scale = 1.0
    initializer = CoarseInitializer(camMat, scale=scale, window_size=2, matches_ratio=0.7)
    params, variables, camMat, initializer = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        use_simplification=config['gaussian_simplification'])

    # 初始化关键帧列表
    keyframe_list = []
    keyframe_time_indices = []

    # 加载检查点（如果有）
    if config['load_checkpoint']:
        # 根据单目SLAM的检查点机制进行调整
        pass
    else:
        checkpoint_time_idx = 0

    # 初始化运行时间记录
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # 主循环
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        print()  # 显示全局迭代

        # 加载当前帧数据
        color, _, _, gt_pose = dataset[time_idx]
        color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)

        # 优化仅针对当前时间步进行跟踪
        iter_time_idx = time_idx
        curr_data = {'cam': camMat, 'im': color, 'id': iter_time_idx, 'intrinsics': camMat, 
                     'iter_gt_w2c_list': None}  # 由于单目SLAM，可能不需要w2c列表

        # 初始化相机位姿（使用单目估计器）
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # 使用单目姿态估计器进行跟踪
            img_np = color.cpu().numpy().transpose(1, 2, 0)
            rst = initializer.forward(img_np)
            if rst is not None:
                pose, points = rst
                # 更新相机位姿
                with torch.no_grad():
                    params['cam_unnorm_rots'][..., time_idx] = torch.tensor(pose[:3, :3]).cuda().float()
                    params['cam_trans'][..., time_idx] = torch.tensor(pose[:3, 3]).cuda().float()
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # 使用真实位姿
                pose = gt_pose
                params['cam_unnorm_rots'][..., time_idx] = torch.tensor(pose[:3, :3]).cuda().float()
                params['cam_trans'][..., time_idx] = torch.tensor(pose[:3, 3]).cuda().float()

        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        # 映射与关键帧选择
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # 映射
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # 添加新高斯
                # 需要基于新的点云生成策略进行调整
                pass

            # 选择关键帧
            with torch.no_grad():
                # 获取当前估计的位姿
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # 选择关键帧（基于位姿变化或其他策略）
                selected_keyframes = keyframe_selection_overlap(matched_points=None,  # 单目SLAM可能不使用重叠选择
                                                                w2c=curr_w2c, 
                                                                intrinsics=None, 
                                                                keyframe_list=keyframe_list[:-1], 
                                                                num_keyframes=config['mapping_window_size']-2)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # 初始化优化器
            optimizer = initialize_optimizer(params, config['mapping']['lrs']) 

            # 映射优化循环
            mapping_start_time = time.time()
            if config['mapping']['num_iters'] > 0:
                progress_bar = tqdm(range(config['mapping']['num_iters']), desc=f"Mapping Time Step: {time_idx}")
            
            for iter in range(config['mapping']['num_iters']):
                iter_start_time = time.time()
                # 随机选择一个关键帧进行优化
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # 使用当前帧数据
                    iter_time_idx = time_idx
                    iter_color = color
                else:
                    # 使用关键帧数据
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']

                # 准备优化数据
                iter_data = {'cam': camMat, 'im': iter_color, 'id': iter_time_idx, 'intrinsics': camMat, 
                             'iter_gt_w2c_list': None}

                # 计算损失
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                  config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                  config['mapping']['use_l1'], tracking=False, 
                                                  mapping=True)

                # 反向传播
                loss.backward()

                # 优化器更新
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # 更新进度条
                if config['report_iter_progress']:
                    report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                    mapping=True, online_time_idx=time_idx)
                else:
                    progress_bar.update(1)

                # 记录运行时间
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1

            if config['mapping']['num_iters'] > 0:
                progress_bar.close()

            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            # 评估映射结果
            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')

        # 添加关键帧到列表
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(params['cam_trans'][..., time_idx]).any()) and (not torch.isnan(params['cam_trans'][..., time_idx]).any()):
            with torch.no_grad():
                # 获取当前估计的旋转和平移
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # 初始化关键帧信息
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': None}
                # 添加到关键帧列表
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

        # 检查点保存
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))

        torch.cuda.empty_cache()

    # 计算平均运行时间
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    with open(os.path.join(output_dir, "runtimes.txt"), "w") as f:
        f.write(f"Average Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms\n")
        f.write(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s\n")
        f.write(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms\n")
        f.write(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s\n")
        f.write(f"Frame Time: {tracking_frame_time_avg + mapping_frame_time_avg} s\n")

    # 评估最终参数
    with torch.no_grad():
        eval_save(dataset, params, eval_dir, sil_thres=config['mapping']['sil_thres'],
                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'])

    # 添加相机参数以保存
    params['timestep'] = variables['timestep']
    params['intrinsics'] = camMat
    params['w2c'] = torch.eye(4).numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    # 由于单目SLAM没有深度信息，可能不需要保存gt_w2c_all_frames
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)

    # 保存参数
    save_params(params, output_dir)
    save_means3D(params['means3D'], output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    # parser.add_argument("--online_vis", action="store_true", help="Visualize mapping renderings while running")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # 设置实验种子
    seed_everything(seed=experiment.config['seed'])
    
    # 创建结果目录并复制配置文件
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)
    
    plot_video(os.path.join(results_dir, 'eval', 'plots'), os.path.join('./experiments/', experiment.group_name, experiment.scene_name, 'keyframes'))
