import json
import logging

import numpy as np
import torch
from sskit.utils import grid2d

from soft_nms import get_detected_circles_from_tensor, soft_non_maximum_suppression

def load_annotations(debug=False, data_dir="./data/", get_val=False):
    if debug:
        path_trn = f'{data_dir}annotations/mini.json'
        path_val = f'{data_dir}annotations/mini.json'
    else:
        path_trn = f'{data_dir}annotations/train.json'
        path_val = f'{data_dir}annotations/val.json'
    with open(path_trn, 'r') as file:
        data_train = json.load(file)
    trn_images = data_train.get("images", [])
    trn_annotations = data_train.get("annotations", [])
    trn_images_dict = {image["id"]: image for image in trn_images}
    trn_annotations_dict = {}
    for annotation in trn_annotations:
        image_id = annotation["image_id"]
        if image_id not in trn_annotations_dict:
            trn_annotations_dict[image_id] = []
        trn_annotations_dict[image_id].append(annotation)
    if not get_val:
        return trn_images_dict, trn_annotations_dict
    else:
        with open(path_val, 'r') as file:
            data_val = json.load(file)
        val_images = data_val.get("images", [])
        val_annotations = data_val.get("annotations", [])
        val_images_dict = {image["id"]: image for image in val_images}
        val_annotations_dict = {}
        for annotation in val_annotations:
            image_id = annotation["image_id"]
            if image_id not in val_annotations_dict:
                val_annotations_dict[image_id] = []
            val_annotations_dict[image_id].append(annotation)
        return trn_images_dict, trn_annotations_dict, val_images_dict, val_annotations_dict

def get_pitch_size(trn_annotations_dict):
    min_pos = [1e10, 1e10]
    max_pos = [-1e10, -1e10]
    for idx in trn_annotations_dict:
        for anno in trn_annotations_dict[idx]:
            pop = anno['position_on_pitch']
            if pop[0] < min_pos[0]:
                min_pos[0] = pop[0]
            if pop[1] < min_pos[1]:
                min_pos[1] = pop[1]
            if pop[0] > max_pos[0]:
                max_pos[0] = pop[0]
            if pop[1] > max_pos[1]:
                max_pos[1] = pop[1]
    pitch_range = np.abs(max_pos) + np.abs(min_pos)
    voxel_grid_size_col = int(pitch_range[1] + 2) # number meters in x direction
    voxel_grid_size_row = int(pitch_range[0] + 2) # number meters in y direction
    return voxel_grid_size_col, voxel_grid_size_row

def create_ground_truth_tensor(model, output, annotations_dict, im_idxs):
    center = - torch.tensor([model.voxel_grid_size_col/2, model.voxel_grid_size_row/2])
    gnd = grid2d(model.voxel_grid_size_col * model.voxel_res, 
                 model.voxel_grid_size_row * model.voxel_res) / model.voxel_res + center
    # Col_max and row_max are on the far side to the left (in the image)
    gnd = torch.flip(gnd, (0, 1, 2))
    gt_io = torch.zeros((output.shape[0], output.shape[2], 
                         output.shape[3])).to(dtype=torch.float64)
    gt_dist = 1e4*torch.ones((output.shape[0], output.shape[2], 
                              output.shape[3], 2)).to(dtype=torch.float32)
    gt_io = gt_io.to(output.device)
    gt_dist = gt_dist.to(output.device)
    for ii, im_i in enumerate(im_idxs):
        annotations = annotations_dict[im_i.item()]
        for annotation in annotations:
            pose = annotation["position_on_pitch"]
            err = (gnd - torch.FloatTensor(pose).reshape(1,1,2)).to(torch.float32).to(output.device)
            diff = torch.norm(err, dim=2)
            inds = torch.where(diff < 0.5)
            pick_ids = torch.norm(gt_dist[ii, inds[0], inds[1], :], dim=1) > torch.norm(err[inds[0], inds[1], :], dim=1)
            gt_dist[ii, inds[0][pick_ids], inds[1][pick_ids], :] = err[inds[0][pick_ids], inds[1][pick_ids], :]
            gt_io[ii, inds[0], inds[1]] = torch.where(gt_io[ii, inds[0], inds[1]] == 0, 1.0, gt_io[ii, inds[0], inds[1]])
    return gt_io, gt_dist, gnd

def create_nms_output(model, output, est_d, sig, im_idxs, use_loss, id_counter, params=(2,20,-10)):
    center = - torch.tensor([model.voxel_grid_size_col/2, model.voxel_grid_size_row/2])
    gnd = grid2d(model.voxel_grid_size_col * model.voxel_res, 
                 model.voxel_grid_size_row * model.voxel_res) / model.voxel_res + center
    # Col_max and row_max are on the far side to the left (in the image)
    gnd = torch.flip(gnd, (0, 1, 2))
    threshold_detection_score = 0.3

    for b in range(output.shape[0]):
        tensor = sig(output[b, 0:1, :, :]).permute(1, 2, 0)
        tensor = torch.cat((tensor, gnd.to(tensor.device), est_d[b,:,:,:]), dim=-1)
        data=[id_counter, im_idxs[b], use_loss]
        detected_circles, id_counter = get_detected_circles_from_tensor(tensor,
                                                                        threshold_detection_score, 
                                                                        params[0], data)
        if detected_circles != []:
            detections=soft_non_maximum_suppression(detected_circles,threshold_detection_score,
                                                    params[1],params[2])
        else:
            detections = []
    return detections, id_counter

def create_logger():
    logger = logging.getLogger("train-log")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("training.log", mode="w")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_training_data(text):
    train_log = logging.getLogger("train-log")
    train_log.info(text)