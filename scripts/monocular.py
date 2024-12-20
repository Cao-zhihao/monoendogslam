# This script is for ...
# Author: ZHANG Yingkui 
# Date: 2024-10-31

import os, sys, glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2, yaml    # pip install opencv-python==4.9.0.80
import pycolmap     # conda install -c conda-forge pycolmap, [Note: please install PyTorch first]
# import pyceres    # pip install pyceres
from typing import Any, List, Dict, Optional, Union
from datetime import datetime
from torch import Tensor, tensor, from_numpy, stack
from matplotlib import pyplot as plt



class FrameData():
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.relative_pose = np.eye(4, dtype=np.float32)

    def __getattr__(self, key):
        if key not in self.__dict__.keys():
            setattr(self, key, None)
        pass

# Usage:
# estimator = CoarseInitializer(camMat, scale=0.5, window_size=2, matches_ratio=0.7)
# rst = estimator.forward(image)
# if rst is not None:
#     pose, points = rst
# ...
class CoarseInitializer():
    def __init__(self, cam_mat: np.ndarray, scale: float, matches_ratio=0.7, triangle_angle=5.0, reprojection_error=0.5, window_size=3):
        self.cam_mat = (cam_mat * scale).astype(np.float32)
        self.scale = scale
        self.matches_ratio = matches_ratio
        # self.triangle_angle = triangle_angle
        # self.reprojection_error = reprojection_error
        self.window_size = window_size

        # 1. feature extracting
        # self.sift_cv2 = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=12, sigma=2.5)
        sift_opt = pycolmap.SiftExtractionOptions(
            edge_threshold=10.0, 
            peak_threshold=1.0/120, 
            dsp_max_scale=3.0, 
            estimate_affine_shape=False, 
            max_num_features=4096,
            normalization='L1_ROOT', # 'L1_ROOT' or 'L2'
            )
        self.sift = pycolmap.Sift(sift_opt, pycolmap.Device.cuda)  # auto, cpu, cuda

        # 2. feature matching, matched points are extracted by employing knn-based matching using FLANN
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # frame window
        self.frame_window = []

        pass

    def preprocess(self, img: Union[np.ndarray, Tensor]):
        assert img.ndim <= 3, "The input [img] shoud be a single image with the max 3 dimensions"

        if isinstance(img, np.ndarray):
            pass
        elif isinstance(img, Tensor):
            img = img.detach().cpu().numpy().squeeze()
        else:
            raise NotImplemented()

        # processing
        if img.ndim == 3:
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # downsampling
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            


        # # 2. LAB channel
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # LAB
        # img2_L = img[..., 0]
        # img2_A = img[..., 1]
        # img2_B = img[..., 2]
        # # 3. CLAHE on L channel
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img2_L_clahe = clahe.apply(img2_L)
        # # 4. merge with AB channel
        # img3 = np.stack([img2_L_clahe, img2_A, img2_B], axis=-1)
        # # 5. convert to RGB
        # img4 = cv2.cvtColor(img3, cv2.COLOR_LAB2RGB)

        # light spot removing
        # _, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # spotMask = np.zeros_like(binary)
        # for cont in contours:
        #     x, y, h, w = cv2.boundingRect(cont)
        #     spotMask[y:y+w, x:x+h] = 255
        # img_rst = cv2.illuminationChange(img, spotMask, alpha=1, beta=2)

        # _, spotMask = cv2.threshold(img_gray, 150, 1, cv2.THRESH_BINARY)
        # spotMask = cv2.dilate(spotMask, np.array([3, 3]), iterations=2)
        # dataMask = 1 - spotMask
        # dataVal = dataMask * img_gray
        # spotVal = int(dataVal.mean()) * spotMask
        # img_rst = dataVal + spotVal

        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # img_gray = clahe.apply(img_gray)

        return img

    def feature_extracting(self, img: Union[np.ndarray, Tensor]):
        # keypoints, descriptors = self.sift_cv2.detectAndCompute(img, None)
        # keypoints = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.response] for kp in keypoints])
        keypoints, descriptors = self.sift.extract(img)
        return keypoints, descriptors

    def feature_matching(self, frame1: FrameData, frame2: FrameData):
        matches = self.flann.knnMatch(frame1.descriptors, frame2.descriptors, k=2)

        # The matches are filtered based on the relative distance and good matches are considered. From the good matches, 'points1' and corresponding 'points2' are extracted.
        # matches_good = list(filter(lambda x: x[0].distance < 0.7 * x[1].distance, matches))
        matches_good = [x[0] for x in matches if (x[0].distance < x[1].distance * self.matches_ratio)]
        # if len(matches_good) > 100:
        #     matches_good = sorted(matches_good, key=lambda x: x.distance)
        #     matches_good = matches_good[:100]

        # select matched points
        matches_idx = np.array([[m.queryIdx, m.trainIdx] for m in matches_good])
        matched1 = frame1.keypoints[matches_idx[:, 0], :2]
        matched2 = frame2.keypoints[matches_idx[:, 1], :2]
        
        if IS_DEBUG:
            points1 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=kp[2]) for kp in frame1.keypoints]
            points2 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=kp[2]) for kp in frame2.keypoints]
            show_img(frame1.img, points1, "detected keypoints of reference image")
            show_img(frame2.img, points2, "detected keypoints of current image")
            img_matches = cv2.drawMatches(frame1.img, points1, frame2.img, points2, matches_good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            show_img(img_matches, title="matches filtered")
        
        return matched1, matched2

    def calc_pose(self, matched1, matched2):
        R, T = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        # calc essential mat via cv2.findEssentialMat
        essentialMat, mask_esse = cv2.findEssentialMat(matched1, matched2, self.cam_mat, cv2.RANSAC, prob=0.999, threshold=0.5)
        mask_esse = mask_esse.squeeze().astype(np.bool_)

        # # another implementation: calc essential mat via pycolmap.essential_matrix_estimation, slower!!!
        # cam = pycolmap.Camera(model=pycolmap.CameraModelId.SIMPLE_RADIAL, width=675, height=540, params=[self.cam_mat[0, 0], self.cam_mat[0, 2], self.cam_mat[1, 2]])
        # ransac_opt = pycolmap.RANSACOptions(max_error=4.0, min_inlier_ratio=0.01, confidence=0.9999, min_num_trials=1000, max_num_trials=100000)
        # essential = pycolmap.essential_matrix_estimation(matched1, matched2, cam, cam, ransac_opt)
        # essentialMat, mask_esse = essential["E"], essential["inliers"]
        # RT = essential["cam2_from_cam1"]  # RT.rotation.matrix(), RT.translation

        matched1 = matched1[mask_esse]
        matched2 = matched2[mask_esse]
        if mask_esse.sum() >= 8:
            # recover pose via cv2.recoverPose
            num, R0, T, mask_pose = cv2.recoverPose(essentialMat, matched1, matched2, self.cam_mat)  # T is normalized

            # recover pose via cv2.decomposeEssentialMat
            R1, R2, T0 = cv2.decomposeEssentialMat(essentialMat)  # T is normalized
            R = R1 if R1.trace() > R2.trace() else R2  # ref: NR-SLAM -> essential_matrix_initialization.cc -> function: ReconstructCameras -> Line 294
            # R_err, T_err = calcError_RT(np.eye(3), np.zeros((3)), R, T)

        return R.astype(np.float32), T.astype(np.float32), matched1, matched2
    
    def sparse_reconstruction(self, R: np.ndarray, T: np.ndarray, matched1, matched2):
        P1 = np.hstack([self.cam_mat, np.zeros((3, 1), dtype=np.float32)])
        P2 = self.cam_mat @ np.hstack([R, T])

        points_3d, points_mask = None, None
        if matched1.shape[0] > 0:
            # points_home = cv2.triangulatePoints(P1, P2, matched1.transpose(), matched2.transpose())
            points_home = cv2.triangulatePoints(P2, P1, matched2.transpose(), matched1.transpose())
            points_mask = points_home[3] != 0
            point_4d = points_home / points_home[3] 
            points_3d = point_4d[:-1].T

        return points_3d, points_mask

    def forward(self, img: Union[np.ndarray, Tensor]):
        # 1. init current frame
        img = self.preprocess(img)
        keypoints, descriptors = self.feature_extracting(img)  # time consumer 1st, 65%
        frame_data = FrameData(keypoints, descriptors)
        if IS_DEBUG:
            frame_data.img = img

        # 2. control frame window
        if len(self.frame_window) == self.window_size:
            self.frame_window.pop(0)

        self.frame_window.append(frame_data)
        rst = None

        # 3. tracking
        if len(self.frame_window) >= 2:
            frame1 = self.frame_window[-2]
            frame2 = self.frame_window[-1]

            matched1, matched2 = self.feature_matching(frame1, frame2)
            R, T, matched1, matched2 = self.calc_pose(matched1, matched2)  # time consumer 2nd, 30%
            points, points_mask = self.sparse_reconstruction(R, T, matched1, matched2)
            
            # to Pose
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, -1] = T.squeeze()
            if points is None: points = frame1.points

            # duplicate when small count
            to_cnt = 3000
            if len(points) < to_cnt:
                multi = int(np.ceil(3000 / len(points)))
                points = np.tile(points, (multi, 1))
                pass

            # save
            frame2.relative_pose = pose
            frame2.points = points

            rst = [pose, points]

            if IS_DEBUG:
                points1 = [cv2.KeyPoint(x=m1[0], y=m1[1], size=1) for m1 in matched1]
                points2 = [cv2.KeyPoint(x=m2[0], y=m2[1], size=1) for m2 in matched2]
                matches = [cv2.DMatch(i, i, -1, -1) for i in range(matched1.shape[0])]
                img_matches = cv2.drawMatches(frame1.img, points1, frame2.img, points2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                show_img(img_matches, title="matches cv2.findEssentialMat masked")
                pass
                
        if IS_DEBUG:
            cv2.destroyAllWindows()

        return rst



def show_imgs(imgs: list):
    imgs = [im.detach().cpu().numpy() if isinstance(im, Tensor) else im for im in imgs]
    imgs = [im.squeeze() for im in imgs]

    cv2.namedWindow('show_img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('show_img', 400 * len(imgs), 300)
    cv2.imshow('show_img', np.hstack(imgs))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

def show_img(img, points=None, title=None):
    if isinstance(img, Tensor):
        img = img.detach().cpu().numpy().squeeze()
    H, W = img.shape[:2]

    if points != None:
        img = cv2.drawKeypoints(img, points, None, color=(0, 255, 0))

    now = datetime.now().strftime("%H:%M:%S.%f")
    info = F"img_size: ({H}, {W})"
    if points is not None:
        info = F"{info}, n_points: {len(points)}"
    title = F"[{now}] [{title}] [{info}]"

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, int(600*W/H), 600)
    cv2.imshow(title, img)
    pass

def save_img(img, filename=None, exist_replace=True):
    if isinstance(img, Tensor):
        img = img.detach().cpu().numpy().squeeze()

    if not filename:
        desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
        filename = os.path.join(desktop_path, "0000_test.jpg")

    counter = 0
    while not exist_replace and os.path.exists(filename):
        basename, extension = os.path.splitext(filename)
        filename = f"{basename}_{counter}{extension}"
        counter += 1

    cv2.imwrite(filename, img)
    print(F"file saved: [{filename}]")
    pass


def calcAngle(dir1, dir2):
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)
    
    cos_theta = (dir1 * dir2).sum()
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_rad = np.arccos(cos_theta)
    angle = np.degrees(theta_rad)
    return angle

def calcError_RT(R1, T1, R2, T2):
    dir = np.random.rand(3, 1)
    dir = dir / np.linalg.norm(dir)
    n1 = R1 @ dir
    n2 = R2 @ dir
    angle = calcAngle(n1, n2)

    dist = np.linalg.norm(T1.squeeze() - T2.squeeze())
    
    # print(F"R error: {angle:.7f}, T error: {dist:.7f} ")
    return angle, dist

def calcError_Pose(Pose1, Pose2):
    R1 = Pose1[:3, :3]
    T1 = Pose1[:-1, -1]
    R2 = Pose2[:3, :3]
    T2 = Pose2[:-1, -1]
    return calcError_RT(R1, T1, R2, T2)

def evalError():
    color_dir = R"D:\2_SIAT\2_code\python\3d\EndoGSLAM\data\EndoOurs\color"
    pose_file = R"D:\2_SIAT\2_code\python\3d\EndoGSLAM\data\EndoOurs\pose.txt"

    # all color files
    filenames = glob.glob(os.path.join(color_dir, "*.png"))
    filenames = sorted(filenames)
    
    # poses
    with open(pose_file, "r") as f:
        lines = f.readlines()
        poses = [np.array(list(map(float, lines[i].split(sep=',')))).reshape(4, 4).transpose() for i in range(len(lines))]

    # calc
    positions1 = []
    positions2 = []
    w2c = np.eye(4, dtype=np.float32)
    w2c2 = np.eye(4, dtype=np.float32)

    # calc
    camMat = np.array([[723.7722, 0, 337.5], [0, 723.7722, 270.0], [0, 0, 1]])
    estimator1 = CoarseInitializer(camMat, 1.0, window_size=2, matches_ratio=0.7)
    estimator2 = CoarseInitializer(camMat, 0.5, window_size=2, matches_ratio=0.7)

    for i in range(len(filenames)):
        img_path = filenames[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE|cv2.IMREAD_IGNORE_ORIENTATION)

        now1 = datetime.now()
        rst1 = estimator1.forward(img)
        rst2 = estimator2.forward(img)
        if rst1 is not None:
            pose1, points1 = rst1
            pose2, points2 = rst2

            # calc position
            w2c = w2c @ pose1
            positions1.append(w2c[:-1, -1])

            w2c2 = w2c2 @ pose2
            positions2.append(w2c2[:-1, -1])

            # calc relative pose
            poses0_ref = poses[i-1]
            poses1_ref = poses[i]
            Rerr_ref, Terr_ref = calcError_Pose(poses0_ref, poses1_ref)

            Rerr_1, Terr_1 = calcError_Pose(np.eye(4), pose1)
            Rerr_2, Terr_2 = calcError_Pose(np.eye(4), pose2)


            print(F"Rerror: {Rerr_ref:.3f}, Terror: {Terr_ref:.3f}")
            print(F"Rerror: {Rerr_1:.3f}, Terror: {Terr_1:.3f}")
            print(F"Rerror: {Rerr_2:.3f}, Terror: {Terr_2:.3f}  \n")

        
        pass

        # time
        now2 = datetime.now()
        # print(F"time1: {(now2-now1).total_seconds()} \n")

        # if IS_DEBUG: 
        #     print(F"The R matrix is: \n{R}", "\n")
        #     print(F"The T matrix is: \n{T}", "\n")

        # M = np.linalg.inv(poses[i]) * poses[i+1]  # TODO test
        # Rgt = M[:3, :3]
        # Tgt = M[-1, :]
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    xs, ys, zs = np.hsplit(np.array(positions1), 3)
    ax.plot(xs.squeeze().tolist(), ys.squeeze().tolist(), zs.squeeze().tolist(), marker='o')

    xs, ys, zs = np.hsplit(np.array(positions2), 3)
    ax.plot(xs.squeeze().tolist(), ys.squeeze().tolist(), zs.squeeze().tolist(), marker='o')

    # 显示图
    plt.show()


    pass

def plotPoses():
    pose_file = R"D:\2_SIAT\2_code\python\3d\EndoGSLAM\data\EndoOurs\pose.txt"

    # poses
    with open(pose_file, "r") as f:
        lines = f.readlines()
        poses = [np.array(list(map(float, lines[i].split(sep=',')))).reshape(4, 4).transpose() for i in range(len(lines))]

    positions = [p[:-1, -1] for p in poses]
    rotmats = [p[:3, :3] for p in poses]

    # 创建一个3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置标签
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')


    xs, ys, zs = np.hsplit(np.array(positions), 3)
    ax.plot(xs.squeeze().tolist(), ys.squeeze().tolist(), zs.squeeze().tolist(), marker='o')


    # # 绘制位置
    # ax.scatter(positions[:][0], positions[:][1], positions[:][2], c='r', marker='o', label='Camera Positions')
    # # 绘制朝向
    # for i in range(len(poses)):
    #     rot = rotmats[i]
    #     direction = rot[:, 2]
    #     # 绘制相机的前向方向（z轴）
    #     ax.quiver(positions[:][0], positions[:][1], positions[:][2], direction[0], direction[1], direction[2], color='b', length=1.0)


    # 显示图
    plt.show()
        
    pass



if __name__ == "__main__":

    IS_DEBUG = getattr(sys, 'gettrace', None) is not None and sys.gettrace() is not None

    mode = cv2.IMREAD_COLOR
    mode = cv2.IMREAD_GRAYSCALE

    evalError()
    plotPoses()


    pass

    # n_iter = 1000
    # now1 = datetime.now()

    # now2 = datetime.now()
    # print(F"time1: {(now2-now1).total_seconds()} \n")
