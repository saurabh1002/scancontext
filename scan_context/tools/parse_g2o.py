#!/bin/python3
import copy
import os

import numpy as np
import open3d as o3d
import typer
from matplotlib import pyplot as plt
from kiss_icp.datasets.generic import GenericDataset
from pgo.pose_graph_optimizer import PoseGraphOptimizer


def get_path_len(poses):
    l = 0
    for i in range(0, len(poses) - 1):
        l += np.linalg.norm(poses[i + 1, :3, -1] - poses[i, :3, -1])
    return l


def plot_poses(poses_gt, poses_pred, poses_opt):
    fig = plt.figure()
    plt.plot(poses_gt[:, 0, -1], poses_gt[:, 1, -1], color="red")
    plt.plot(poses_pred[:, 0, -1], poses_pred[:, 1, -1], color="tab:blue")
    plt.plot(poses_opt[:, 0, -1], poses_opt[:, 1, -1], c="lime")
    plt.legend(["groundtruth", "kiss-icp", "g2o optimized"])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.show()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556],
    )


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def gicp(source, target, initial_guess, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: GICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        distance_threshold,
        initial_guess,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    )
    return result


def global_registration(source, target, voxel_size):
    s, desc_s = preprocess_point_cloud(source, voxel_size)
    t, desc_t = preprocess_point_cloud(target, voxel_size)
    ransac_estimate = ransac(s, t, desc_s, desc_t, voxel_size)
    estimate = gicp(s, t, ransac_estimate.transformation, voxel_size)
    if estimate.fitness > 0.9:
        return True, estimate.transformation
    else:
        return False, np.eye(4)


def main(data_dir: str = typer.Argument(""), gt_poses_file: str = typer.Argument(""), closure_dir: str = typer.Argument("")):
    optimizer = PoseGraphOptimizer()
    dataset = GenericDataset(data_dir)
    # Load poses and add them to the graph with the odometry edges
    gt_poses = np.load(gt_poses_file)

    pose_file = os.path.join(closure_dir, "poses.npy")
    poses = np.load(pose_file)
    for idx, pose in enumerate(poses):
        optimizer.add_variable(idx, pose)
    omega = np.eye(6)
    for idx in range(len(poses) - 1):
        Ti = poses[idx]
        Tj = poses[idx + 1]
        optimizer.add_factor(idx, idx + 1, np.linalg.inv(Ti) @ Tj, omega)

    closures_file = os.path.join(closure_dir, "closures.txt")
    closures = np.loadtxt(closures_file)
    if len(closures.shape) == 1:
        closures = closures.reshape(1, -1)
    omega_closure = 1e3 * np.eye(6)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        for ids in closures:
            try:
                pose = ids[2:].reshape(4, 4)
            except ValueError:
                continue
            scan_i = int(ids[0])
            scan_j = int(ids[1])
            source = dataset[scan_i]
            target = dataset[scan_j]
            success = False
            estimate = gicp(source, target, pose, 1.0)
            if estimate.fitness > 0.5:
                success = True
            if success:
                # draw_registration_result(source, target, estimate.transformation)
                optimizer.add_factor(scan_j, scan_i, estimate.transformation, omega_closure)

        optimizer.optimize()
        optimizer.write_graph(os.path.join(closure_dir, "out.g2o"))
        poses_map = dict(optimizer.estimates())

        poses_opt = np.ones((len(poses_map), 4, 4))
        for i, pose_opt in poses_map.items():
            poses_opt[i] = pose_opt

        poses_opt = np.einsum("ij,njk->nik", np.linalg.inv(poses_opt[0]), poses_opt)
        det = np.linalg.det(poses_opt[:, :3, :3])
        print(np.average(det))
        print(get_path_len(gt_poses), get_path_len(poses), get_path_len(poses_opt))
        print(len(gt_poses), len(poses), len(poses_opt))
        plot_poses(gt_poses, poses, poses_opt)
        np.save(os.path.join(closure_dir, "optimized_poses.npy"), poses_opt)
        np.savetxt(os.path.join(closure_dir, "gt_poses_kitti.txt"), gt_poses[:, :3].reshape(-1, 12))
        np.savetxt(
            os.path.join(closure_dir, "optimized_poses_kitti.txt"), poses_opt[:, :3].reshape(-1, 12)
        )


if __name__ == "__main__":
    typer.run(main)
