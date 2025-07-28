import json
import os
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
from clearml import Task, Logger
from sskit.coco import LocSimCOCOeval
from xtcocotools.coco import COCO

from network import SimpleBEVNetworkWithResNet
from dataset import BEVDataset
from train import run_batch
from utils import load_annotations, get_pitch_size, log_training_data, create_logger

def main(data_dir, save_dir, cont, debug, num_epochs, params, backbone, batch_size, eval_only=False):
    (trn_images_dict,
     trn_annotations_dict,
     val_images_dict,
     val_annotations_dict) = load_annotations(debug=debug, data_dir=data_dir, get_val=True)
    voxel_grid_size_col, voxel_grid_size_row = get_pitch_size(trn_annotations_dict)

    # Create dataset and data loader
    if debug:    
        trn_dataset = BEVDataset(trn_images_dict, image_dir=f'{data_dir}mini/')
        val_dataset = BEVDataset(val_images_dict, image_dir=f'{data_dir}mini/')
    else:
        trn_dataset = BEVDataset(trn_images_dict, image_dir=f'{data_dir}train/')
        val_dataset = BEVDataset(val_images_dict, image_dir=f'{data_dir}val/')
    num_workers = 2*batch_size if 2*batch_size < 16 else 16
    num_workers = 0
    trn_data_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #TODO
    device = torch.device("cpu")

    # Instantiate the model, loss function, and optimizer
    voxel_grid_size = [voxel_grid_size_row, voxel_grid_size_col, 3]
    if backbone == "unet":
        model = SimpleBEVNetworkWithResNet(feature_extractor=backbone, res_net_tail=False,
                                           voxel_grid_size=voxel_grid_size,                                      
                                           feature_channels=params[0], unet_features=params)
    else:
        model = SimpleBEVNetworkWithResNet(feature_extractor=backbone, res_net_tail=False,
                                           voxel_grid_size=voxel_grid_size, feature_channels=params)
    if cont is not None:
        model.load_state_dict(torch.load(cont))
        log_training_data("Successfully loaded parameters")
    assert os.path.exists("BCE_weight.npy"), "No weight file found, run preliminaries first"
    log_training_data("Loading bce_weight")
    weight = torch.as_tensor(np.load("BCE_weight.npy"))
    log_training_data("Done!")
    loss_fkt = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    sig = torch.nn.Sigmoid()

    # Training loop
    model.to(device)
    all_trn_losses = []
    all_val_losses = []
    best_loss = 1e4
    log_training_data("Training starting")

    for epoch in range(num_epochs):
        # ============== Training ========================
        model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        epoch_loss = 0.0
        training_pos_est = []
        id_counter = 0

        for batch_idx, (image_vals, dist_poly, camera_matrix, im_idxs, file_name) in enumerate(trn_data_loader):
            output, gt_io, e_loss, est_dict, _ = run_batch(debug, True, device, optimizer, model,
                image_vals, dist_poly, camera_matrix, im_idxs, loss_fkt, trn_annotations_dict, sig,
                file_name, running_loss, id_counter=id_counter)
            training_pos_est.extend(est_dict)
            id_counter = len(training_pos_est)
            epoch_loss += sum(e_loss)
            running_loss += sum(e_loss)
            running_loss1 += e_loss[0]
            running_loss2 += e_loss[1]
            bs_print = 30
            if batch_idx % bs_print == (bs_print - 1):
                log_training_data(f"Epoch {epoch+1}, Batch {batch_idx} of {len(trn_data_loader)} Loss: {running_loss / bs_print:.4f}")
                Logger.current_logger().report_scalar("train", "loss1",
                                                      iteration=(epoch * len(trn_data_loader) + batch_idx),
                                                      value=running_loss1/bs_print)
                Logger.current_logger().report_scalar("train", "loss2",
                                                      iteration=(epoch * len(trn_data_loader) + batch_idx),
                                                      value=running_loss2/bs_print)
                running_loss = 0.0
                running_loss1 = 0.0
                running_loss2 = 0.0
                torch.save(model.state_dict(), f"./models/current_{model.model_name}.pt")
        if epoch % 5 == 0:
            #Make confusion matrix
            confusion_matrix = torch.zeros(2, 2)
            confusion_matrix[0, 0] = torch.sum(gt_io * (sig(output[:, 0, :, :]) > 0.5))
            confusion_matrix[0, 1] = torch.sum(gt_io * (sig(output[:, 0, :, :]) < 0.5))
            confusion_matrix[1, 0] = torch.sum((1 - gt_io) * (sig(output[:, 0, :, :]) > 0.5))
            confusion_matrix[1, 1] = torch.sum((1 - gt_io) * (sig(output[:, 0, :, :]) < 0.5))
            log_training_data(f"Confusion matrix (TP, FN \ FP, TN):")
            log_training_data(f"T\E, 1,   0")
            log_training_data(f"1   {confusion_matrix[0,0]}, {confusion_matrix[0,1]}")
            log_training_data(f"0   {confusion_matrix[1,0]}, {confusion_matrix[1,1]}")
        all_trn_losses.append(epoch_loss / len(trn_data_loader))
        # ============== validation ========================
        model.eval()
        val_epoch_loss = 0.0
        validation_pos_est = []
        id_counter = 0

        for batch_idx, (image_vals, dist_poly, camera_matrix, im_idxs, file_name) in enumerate(val_data_loader):
            output, gt_io, e_loss, val_dict, id = run_batch(debug, False, device, optimizer, model, image_vals, dist_poly, camera_matrix, im_idxs, 
              loss_fkt, val_annotations_dict, sig, file_name, id_counter=id_counter)
            val_epoch_loss += sum(e_loss)
            validation_pos_est.extend(val_dict)
            id_counter = id
        all_val_losses.append(val_epoch_loss / len(val_data_loader))
        tmp = {"images": list(val_images_dict.values()),
               "annotations": validation_pos_est,
               "categories": [{"id": 1, "name": "person"}]}
        os.makedirs(f"{save_dir}", exist_ok=True)
        with open(f"{save_dir}{model.model_name}.json", "w") as json_file:
            json.dump(tmp, json_file, indent=4)
        est_data = COCO(annotation_file= "./", ann_data=tmp)
        if debug:
            path_val = f'{data_dir}annotations/mini.json'
        else:
            path_val = f'{data_dir}annotations/val.json'
        with open(path_val, 'r') as file:
            data_val = json.load(file)
        gt_data = COCO(annotation_file="./", ann_data=data_val)
        eval = LocSimCOCOeval(gt_data, est_data, 'bbox', [0.089, 0.089], True)
        eval.params.useSegm = None
        # coco_eval.params.position_from_keypoint_index = 1
        print(f"Validation results on epoch: {epoch}")
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        print(f"mAP LocSim: {eval.stats[0]}")
        print(f"Precision: {eval.stats[12]}")
        print(f"Recall: {eval.stats[13]}")
        print(f"F1: {eval.stats[14]}")
        print(f"Frame Accuracy: {eval.stats[16]}")
        print(f"Score threshold: {eval.stats[15]}")
            
        if (val_epoch_loss / len(val_data_loader)) < best_loss:
            os.makedirs(f"./models/", exist_ok=True)
            torch.save(model.state_dict(), f"./models/model_{model.model_name}.pt")
            best_loss = val_epoch_loss / len(val_data_loader)
        log_training_data(f"End of Epoch {epoch} loss: {epoch_loss / len(trn_data_loader)}")
        log_training_data(f"\tValidation loss: {val_epoch_loss / len(val_data_loader)}")
        Logger.current_logger().report_scalar("Whole epoch", "train-loss",
                                              iteration=(epoch),
                                              value=epoch_loss / len(trn_data_loader))
        Logger.current_logger().report_scalar("Whole epoch", "val-loss",
                                              iteration=(epoch),
                                              value=val_epoch_loss / len(val_data_loader))
    log_training_data("Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_mode", help = "Runs code in debug mode",
                       action="store_true")
    parser.add_argument("-f", "--data_directory", help = "Directory where data is found")
    parser.add_argument("-s", "--save_directory", help = "Directory where data and results are saved")
    parser.add_argument("-b", "--backbone", help = "Which backbone to use")
    parser.add_argument("-p", "--parameters", help = "Number of features used in unet")
    parser.add_argument("-e", "--epochs", help = "Number of Epochs to train")
    parser.add_argument("-bs", "--batch_size", help = "Batch size when training")
    parser.add_argument("-e", "--eval_only", help = "Set to true if only evaluation is to be run",
                        action="store_true")
    parser.add_argument("-c", "--continue_train", help = "Continues to run training from saved model with the name provided")
    parser.add_argument("-a", "--alvis_train", help = "Set flag if run on Alvis HPC, log is saved in personal folder",
                       action="store_true")
    args = parser.parse_args()
    debug = True #args.debug_mode
    cont = args.continue_train
    backbone = args.backbone
    if backbone is None:
        backbone = ""
    hpc = args.alvis_train
    params = args.parameters
    epochs = 1#args.epochs
    if epochs is None:
        epochs = 10
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 1
    else:
        batch_size = int(batch_size)
    if params is not None:
        params = params.split(",")
        params = [int(p) for p in params]
    data_dir = args.data_directory
    save_dir = args.save_directory
    eval_only = args.eval_only
    task = Task.init(project_name="FootballBEV", task_name=f"Training-{backbone}-{params}")

    create_logger()
    log_training_data(f"Runs training - debug: {debug}, continue: {cont},\n\tdata_dir: {data_dir}")
    log_training_data(f"Epochs: {epochs}, parameters: {params}")
    main(data_dir, save_dir, cont, debug, epochs, params, backbone, batch_size, eval_only)
