import torch
import numpy as np


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where((gt_depth[0] > 0) & (gt_depth[0] < 1e10)) # Remove invalid depth values
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    sampled_indices = valid_depth_indices[indices]

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]
        # Filter out the points that are outside the image
        edge = 20
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)
        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # Select the keyframes with percentage of points inside the image > 0
    selected_keyframe_list = [keyframe_dict['id']
                                for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
    selected_keyframe_list = list(np.random.permutation(
        np.array(selected_keyframe_list))[:k])

    return selected_keyframe_list
    
    
def keyframe_selection_distance(
    time_idx, curr_position, prev_position, keyframe_list, 
    distance_current_frame_prob, n_samples):
    """
    选择关键帧，并根据距离、时间和运动方向计算概率。
    """
    distances = []
    time_laps = []
    curr_shift = torch.norm(curr_position - prev_position).item()

    for keyframe in keyframe_list:
        est_w2c = keyframe['est_w2c'].detach().cpu().numpy()
        camera_position = est_w2c[:3, 3]

        # 将 camera_position 转换为 Tensor
        camera_position_tensor = torch.from_numpy(camera_position).to(curr_position.device)
        
        # 计算当前帧与关键帧之间的距离
        distance = torch.norm(camera_position_tensor - curr_position).item()
        time_lap = time_idx - keyframe['id']

        distances.append(distance)
        time_laps.append(time_lap)

    # 使用距离和时间计算概率
    dis2prob = lambda x, scaler: np.log2(1 + scaler / (x + scaler / 5))
    dis_prob = [
        dis2prob(d, curr_shift) + dis2prob(t, time_idx) 
        for d, t in zip(distances, time_laps)
    ]
    
    sum_prob = sum(dis_prob) / (1 - distance_current_frame_prob)
    norm_dis_prob = [p / sum_prob for p in dis_prob]
    norm_dis_prob.append(distance_current_frame_prob)

    # 计算 CDF 并采样
    cdf = np.cumsum(norm_dis_prob)
    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)

    return sample_indices

