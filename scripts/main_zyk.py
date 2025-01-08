import os, sys, time, shutil
IS_DEBUG = getattr(sys, 'gettrace', None) is not None and sys.gettrace() is not None
import argparse
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path: print(p)

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
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, matrix_to_rot
from utils.vis_utils import plot_video
from utils.time_helper import Timer

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from monocular import CoarseInitializer



def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["endoslam_unity"]:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    
    
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    # Image.fromarray(np.uint8((torch.permute(color, (1, 2, 0)) * mask.reshape(320, 320, 1)).detach().cpu().numpy()*255), 'RGB').save('gaussian.png')

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=True):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
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


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None, use_simplification=True):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=use_simplification)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)

    mask = (depth > 0) & energy_mask(color) # Mask out invalid depth values
    # Image.fromarray(np.uint8(mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio # NOTE: change_here

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam, None, None


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, 
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba: # Bundle Adjustment
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    
    # Visualize the Rendered Images
    # online_render(curr_data, iter_time_idx, rendervar, dev_use_controller=False)
        
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D'] # Gradient only accum from colour render for densification

    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    elif tracking:
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


def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def calc_w2c(params, time_idx):
    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
    curr_w2c = torch.eye(4).cuda().float()
    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
    curr_w2c[:3, 3] = curr_cam_tran
    return curr_w2c


def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_w2c = calc_w2c(params, time_idx)

        valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        valid_color_mask = energy_mask(curr_data['im']).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)        
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def update_config(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

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
        dataset_config["gradslam_data_cfg"] = {}
        dataset_config["gradslam_data_cfg"]["dataset_name"] = dataset_config["dataset_name"]
    else:
        dataset_config["gradslam_data_cfg"] = load_dataset_config(dataset_config["gradslam_data_cfg"])
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
        dataset_config["seperate_densification_res"] = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            dataset_config["seperate_densification_res"] = True
        else:
            dataset_config["seperate_densification_res"] = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        dataset_config["seperate_tracking_res"] = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            dataset_config["seperate_tracking_res"] = True
        else:
            dataset_config["seperate_tracking_res"] = False

    return config


def prepare_dataset(config: dict):
    dataset_config = config["data"]
    gradslam_data_cfg = dataset_config['gradslam_data_cfg']
    seperate_tracking_res = dataset_config['seperate_tracking_res']
    seperate_densification_res = dataset_config['seperate_densification_res']
    # Get Device
    device = torch.device(config["primary_device"])

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
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

    eval_dataset = None
    if dataset_config["train_or_test"] == 'train': # kind of ill implementation here. train_or_test should be 'all' or 'train'. If 'test', you view test set as full dataset.
        eval_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"], # if you eval, you should keep reso as raw image.
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test='test')

    # Init seperate dataloader for densification if required
    densify_dataset = None
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"])

    # Init seperate dataloader for tracking if required
    tracking_dataset = None
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"])

    return dataset, eval_dataset, densify_dataset, tracking_dataset



def init_pose_intrinsics(dataset):
    color, _, intrinsics, pose = dataset[0]
    
    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=True)

    return cam, intrinsics


def init_params_zyk(init_pt_cld, w2c, num_frames, intrinsics, time_idx, first_init=True):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]
    
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
    colors = np.tile([0.5, 0.5, 0.5], (num_pts, 1)) # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")


    pts4 = torch.cat((init_pt_cld, torch.ones_like(init_pt_cld[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2]
    scale_gaussian = depth_z / ((FX + FY)/2)
    mean3_sq_dist = scale_gaussian ** 2
    # mean3_sq_dist = np.tile([0.0006], (num_pts, ))
    # mean3_sq_dist = torch.from_numpy(mean3_sq_dist)

    params = {
        'means3D': means3D,
        'rgb_colors': colors,
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }

    if first_init:
        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1)).astype(np.float32)
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        params['cam_unnorm_rots'] = cam_rots  # 1 x 4 x num_frames
        params['cam_trans'] = np.zeros((1, 3, num_frames))  # 1 x 3 x num_frames
        
        params['cam_unnorm_rots'][..., time_idx] = matrix_to_rot(w2c[:3, :3]).cpu().numpy()
        params['cam_trans'][..., time_idx] = w2c[:3, -1].cpu().numpy()

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def update_params_zyk(params, variables, curr_data, init_pt_cld, num_frames, intrinsics, time_idx):
    cam_pose = "cam_unnorm_rots" not in params or "cam_trans" not in params
    params_new = init_params_zyk(init_pt_cld, curr_data['w2c'], num_frames, intrinsics, time_idx, cam_pose)

    for k, v in params_new.items():
        if k in params:
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        else:
            params[k] = params_new[k]

    num_pts = params['means3D'].shape[0]
    variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
    variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
    variables['denom'] = torch.zeros(num_pts, device="cuda").float()

    new_timestep = time_idx * torch.ones(init_pt_cld.shape[0], device="cuda").float()
    if 'timestep' not in variables:
        variables['timestep'] = new_timestep
    else:
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep), dim=0)
    variables['scene_radius'] = params['log_scales'].max() * 5  # parameter

    return params, variables


def tracking(config: dict, params, variables, tracking_curr_data, time_idx):
    loss_min = 1e10

    if time_idx > 0:
        # Reset Optimizer & Learning Rates for tracking
        optimizer = initialize_optimizer(params, config['tracking']['lrs'])
        # Keep Track of Best Candidate Rotation & Translation
        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
        current_min_loss = float(1e20)
        # Tracking Optimization
        iter = 0
        num_iters_tracking = config['tracking']['num_iters']
        progress_bar = tqdm(range(num_iters_tracking), ncols=100, leave=False, desc=f"Tracking Time Step: {time_idx}")
        while True:
            # Loss for current frame
            loss, variables, losses = get_loss(params, tracking_curr_data, variables, time_idx, config['tracking']['loss_weights'],
                                            config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                            config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                            visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                            tracking_iteration=iter)
            # Backprop
            loss.backward()
            loss_min = min(loss_min, loss.item())
            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                # Save the best candidate rotation & translation
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                    candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                # Report Progress
                if config['report_iter_progress']:
                    report_progress(params, tracking_curr_data, iter+1, progress_bar, time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                else:
                    progress_bar.update(1)
            # Check if we should stop tracking
            iter += 1
            if iter == num_iters_tracking:
                break

        progress_bar.close()
        # Copy over the best candidate rotation & translation
        with torch.no_grad():
            params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
            params['cam_trans'][..., time_idx] = candidate_cam_tran
    
    return loss_min


def mapping(config: dict, params, variables, curr_data, time_idx, keyframe_list):
    loss_min = 1e10

    num_iters_mapping = config['mapping']['num_iters']

    color = curr_data['im']
    depth = curr_data['depth']
    cam = curr_data['cam']
    intrinsics = curr_data['intrinsics']
    first_frame_w2c = curr_data['w2c']

    # Reset Optimizer & Learning Rates for Full Map Optimization
    optimizer = initialize_optimizer(params, config['mapping']['lrs']) 

    if num_iters_mapping > 0:
        progress_bar = tqdm(range(num_iters_mapping), ncols=100, leave=False, desc=f"Mapping Time Step: {time_idx}")
        
    actural_keyframe_ids = []
    for iter in range(num_iters_mapping):
        if len(actural_keyframe_ids) == 0:
            if len(keyframe_list) > 0:
                curr_position = params['cam_trans'][..., time_idx].detach().cpu()
                actural_keyframe_ids = keyframe_selection_distance(time_idx, curr_position, keyframe_list, config['distance_current_frame_prob'], num_iters_mapping)
            else:
                actural_keyframe_ids = [0] * num_iters_mapping
            print(f"\nUsed Frames for mapping at Frame {time_idx}: {[keyframe_list[i]['id'] if i != len(keyframe_list) else 'c' for i in actural_keyframe_ids]}")

        selected_keyframe_ids = actural_keyframe_ids[iter]

        if selected_keyframe_ids == len(keyframe_list):
            # Use Current Frame Data
            iter_time_idx = time_idx
            iter_color = color
            iter_depth = depth
        else:
            # Use Keyframe Data
            iter_time_idx = keyframe_list[selected_keyframe_ids]['id']
            iter_color = keyframe_list[selected_keyframe_ids]['color']
            iter_depth = keyframe_list[selected_keyframe_ids]['depth']
            
        iter_data = {'cam': cam, 'intrinsics': intrinsics, 
                     'id': iter_time_idx, 'im': iter_color, 'depth': iter_depth,
                     'w2c': first_frame_w2c}
        # Loss for current frame
        loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                        config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                        config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
        # Backprop
        loss.backward()
        loss_min = min(loss_min, loss.item())
        with torch.no_grad():
            # Prune Gaussians
            if config['mapping']['prune_gaussians']:
                params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
            # Gaussian-Splatting's Gradient-based Densification
            if config['mapping']['use_gaussian_splatting_densification']:
                params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Report Progress
            if config['report_iter_progress']:
                report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                mapping=True, online_time_idx=time_idx)
            else:
                progress_bar.update(1)
    
    if num_iters_mapping > 0:
        progress_bar.close()
        
    return loss_min


def mono_slam(config: dict):
    # timer = Timer()
    # timer.start()
    
    config = update_config(config)
    dataset_config = config["data"]

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # dataset
    dataset, eval_dataset, densify_dataset, tracking_dataset = prepare_dataset(config)
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
    num_cascade_iters = config['num_cascade_iters']

    # camera params
    cam, intrinsics = init_pose_intrinsics(dataset)
    coarser = CoarseInitializer(intrinsics.cpu().numpy(), 1, 0.9)

    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    params, variables = {}, {}
    
    # timer.lap("all the config")
    
    # Iterate over Scan
    last_w2c = torch.eye(4, device=intrinsics.device)
    desc = "MonoSLAM - frame:{:>3}       tracking_min_loss: {:.4f}  mapping_min_loss: {:.4f} "
    bar = "{desc}  ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}]"
    pbar = tqdm(total=num_frames, ncols=120, leave=True, desc=desc.format(0, 0, 0), bar_format=bar)

    for time_idx in range(0, num_frames):
        # timer.lap("iterating over frame "+str(time_idx), 0)
        tqdm.write("--"*60)

        # 1. Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        relative_pose, init_pt_cld = coarser.forward(color)
        relative_pose = torch.from_numpy(relative_pose).to(color.device)
        if init_pt_cld is not None: init_pt_cld = torch.from_numpy(init_pt_cld).to(color.device)

        gt_w2c = torch.linalg.inv(gt_pose)
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)

        if IS_DEBUG and False:
            init_pt_cld_d, mean3_sq_dist_d = get_pointcloud(color, depth, intrinsics, gt_w2c, 
                                                            compute_mean_sq_dist=True)

        # 2. init parameters
        w2c = last_w2c @ relative_pose
        curr_data = {'cam': cam, 'intrinsics': intrinsics, 
                     'id': time_idx, 'im': color, 'depth': depth, 
                     'w2c': w2c, 'gt_w2c': gt_w2c, 'last2curr': relative_pose}

        if init_pt_cld is not None:
            params, variables = update_params_zyk(params, variables, curr_data, init_pt_cld, num_frames, intrinsics, time_idx)

        if time_idx == 0:
            pbar.update(1)
            continue

        tracking_min = 1e10
        mapping_min = 1e10
        for cas_idx in tqdm(range(num_cascade_iters), ncols=100, leave=False):
            # 3. Tracking
            tracking_loss = tracking(config, params, variables, curr_data, time_idx)
            tracking_min = min(tracking_loss, tracking_min)

            # 4. Mapping
            mapping_loss = mapping(config, params, variables, curr_data, time_idx, keyframe_list)
            mapping_min = min(mapping_loss, mapping_min)
            
        # 5. Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or (time_idx == num_frames-2)):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_w2c = calc_w2c(params, time_idx)

                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

        torch.cuda.empty_cache()

        pbar.desc = desc.format(time_idx, tracking_min, mapping_min)
        pbar.update(1)
    pbar.close()
    tqdm.write("=="*60 + "\n")

    # Evaluate Final Parameters
    dataset = [dataset, eval_dataset, 'C3VD'] if dataset_config["train_or_test"] == 'train' else dataset
    with torch.no_grad():
        eval_save(dataset, params, eval_dir, sil_thres=config['mapping']['sil_thres'],
                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    # params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    # for gt_w2c_tensor in gt_w2c_all_frames:
    #     params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)
    save_means3D(params['means3D'], output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("experiment", type=str, help="Path to experiment file")
    # parser.add_argument("--online_vis", action="store_true", help="Visualize mapping renderings while running")

    args = parser.parse_args()


    # ZYK
    args.experiment = R"D:\2_SIAT\2_code\python\3d\monoendogslam\configs\c3vd\c3vd_base.py"



    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()

    # Prepare dir for visualization
    # if args.online_vis:
    #     vis_dir = './online_vis'
    #     os.makedirs(vis_dir, exist_ok=True)
    #     for filename in os.listdir(vis_dir):
    #         os.unlink(os.path.join(vis_dir, filename))

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(experiment.config["workdir"], experiment.config["run_name"])
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    mono_slam(experiment.config)
    
    plot_video(os.path.join(results_dir, 'eval', 'plots'), os.path.join('./experiments/', experiment.group_name, experiment.scene_name, 'keyframes'))
