import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sskit.utils import grid2d
from sskit import unnormalize, world_to_image

from utils import load_annotations, get_pitch_size

def calc_pos_hist(debug=False, data_dir="./data/", save_dir="./data_illustration/"):
    _, trn_annotations_dict = load_annotations(debug, data_dir, get_val=False)

    min_dist = 1e4
    all_pos = []
    for key in trn_annotations_dict.keys():
        pos = []
        for ann in trn_annotations_dict[key]:
            pos.append(ann["position_on_pitch"])
        if len(pos) < 2:
            continue
        pos = np.array(pos)
        differences = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # Shape: (N, N, 2)
        diff_m = np.linalg.norm(differences, axis=2)
        md = np.min(diff_m[diff_m > 0])
        tmp = diff_m[np.tril_indices(diff_m.shape[0])]
        all_pos.extend(tmp[tmp > 0])
        if md < min_dist:
            min_dist = md
    plt.figure()
    plt.hist(all_pos, bins=400)
    plt.savefig(f"{save_dir}pos_hist.png")

def calculate_BCE_weight(debug, voxel_res, data_dir="./data/"):
    _, trn_annotations_dict = load_annotations(debug, data_dir, get_val=False)
    voxel_grid_size_col, voxel_grid_size_row = get_pitch_size(trn_annotations_dict)
    ones = 0
    zeros = 0
    center = - torch.tensor([voxel_grid_size_col/2, voxel_grid_size_row/2])
    gnd = grid2d(voxel_grid_size_col * voxel_res, 
                 voxel_grid_size_row * voxel_res) / voxel_res + center
    # Col_max and row_max are on the far side to the left (in the image)
    gnd = torch.flip(gnd, (0, 1, 2))
    for idx in trn_annotations_dict:
        gt_io = torch.zeros((1, gnd.shape[0], gnd.shape[1]))
        for anno in trn_annotations_dict[idx]:
            pose = anno["position_on_pitch"]
            err = (gnd - torch.FloatTensor(pose).reshape(1,1,2)).to(torch.float32)
            diff = torch.norm(err, dim=2)
            inds = torch.where(diff < 0.5)
            gt_io[0, inds[0], inds[1]] = 1
        ones += torch.sum(gt_io)
        zeros += torch.numel(gt_io) - torch.sum(gt_io)
    np.save("BCE_weight.npy", zeros/ones)
    return zeros/ones

def show_image_with_voxels(save=True, data_dir="./data/", save_dir="./data_illustration/", z=0):
    images, trn_annotations_dict = load_annotations(True, data_dir, get_val=False)
    voxel_grid_size_col, voxel_grid_size_row = get_pitch_size(trn_annotations_dict)
    resolution = 1 # number of points per meter
    for id_im, _ in enumerate(images):
        file_name = images[id_im]['file_name']
        image_path = f'{data_dir}/mini/{file_name}'
        camera_matrix = images[id_im]['camera_matrix']
        dist_poly = images[id_im]['dist_poly']
        image = cv2.imread(image_path)
        xv, yv = np.meshgrid(np.arange(voxel_grid_size_row*resolution) / resolution,
                             np.arange(voxel_grid_size_col*resolution) / resolution,
                             indexing="ij")
        xv -= voxel_grid_size_row / 2
        yv -= voxel_grid_size_col / 2
        voxel_grid = np.stack((xv, yv)).reshape(2,-1).T
        voxel_grid = np.concatenate((voxel_grid, z*np.ones((voxel_grid.shape[0], 1))), axis=1)
        image_shape = [image.shape[2], image.shape[0], image.shape[1]]
        image_points = unnormalize(world_to_image(np.array(camera_matrix), np.array(dist_poly), voxel_grid), image_shape)
        for point in image_points:
            inside = (point[0] >= 0) & (point[0] <= image_shape[2]) & \
                        (point[1] >= 0) & (point[1] <= image_shape[1])
            if inside:
                cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), 4)
        ori = np.array([0., 0., 0.])
        origin = unnormalize(world_to_image(np.array(camera_matrix), np.array(dist_poly), ori), image_shape)
        cv2.circle(image, (int(origin[0]), int(origin[1])), 4, (0, 255, 255), 10)
        for anno in trn_annotations_dict[id_im]:
            if anno['image_id'] == id_im:
                pop = anno['position_on_pitch']
                tmp = np.array([pop[0], pop[1], 0])
                im_pop = unnormalize(world_to_image(np.array(camera_matrix),
                                                    np.array(dist_poly),
                                                    tmp),
                                                    image_shape)
                cv2.circle(image, (int(im_pop[0]), int(im_pop[1])), 2, (255, 0, 0), 2)
        scale_percent = 40
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if save:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}image_with_voxels_{id_im}_z_{z}.png", resized_image)
        else:
            cv2.imshow(f"Image with Voxels {id_im}", resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_mode", help = "Runs code in debug mode",
                       action="store_true")
    parser.add_argument("-p", "--data_path", help = "Path to directory where data is found")
    parser.add_argument("-s", "--save_path", help = "Path to directory where data is saved")
    args = parser.parse_args()
    debug = args.debug_mode
    data_path = args.data_path
    save_dir = args.save_path
    calc_pos_hist(debug=debug, data_dir=data_path, save_dir=save_dir)
    calculate_BCE_weight(debug, 2, data_dir=data_path)
    show_image_with_voxels(True, data_path, save_dir, z=0)