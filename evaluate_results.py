import argparse
import json
import os

import torch
from torch.ao.quantization import convert, prepare, get_default_qconfig
from torch.utils.data import DataLoader
import cv2
import numpy as np
from sskit import project_on_ground, imread
from sskit.utils import grid2d
import copy
from thop import profile

from utils import load_annotations, wrap_named_module_as_float_island
from train import run_batch
from network import BEVNetwork
from dataset import BEVDataset

def create_detections_from_tensor(tensor, threshold_detection_score=0.5, radius_detection_circle=0.5, data=None):
    """
    Create a dictionary with coordinates and detection circle radius where the first channel is over a given threshold.
    The x and y values are taken from the second and third channels, respectively.
    
    Parameters:
        tensor (torch.Tensor): A tensor of shape (M, N, 3), where the second channel contains x values 
                               and the third channel contains y values.
        threshold_detection_score (float): The threshold for detection in the first channel. Default is 0.5.
        radius_detection_circle (float): The radius of the detection circle. Default is 0.5.
    
    Returns:
        dict: A dictionary with keys as indices (x, y) and values as the radius for each detection circle.
    """
    # Initialize the dictionary to hold detections
    if data is None:
        detected_circles = {}
        counter = 0
    else:
        detected_circles = []
        id_counter = data[0]  
        use_loss = data[2]  

    # Get the dimensions of the tensor
    M, N, _ = tensor.shape
    
    # Iterate over all coordinates in the tensor
    for i in range(M):
        for j in range(N):
            score = tensor[i, j, 0].item()  # Get the detection score from the first channel
            if score > threshold_detection_score:  # Check if the first channel is above threshold
                
                # Get the x and y coordinates from the second and third channels
                x_coord = tensor[i, j, 1].item() # Second channel (x-coordinate)
                y_coord = tensor[i, j, 2].item() # Third channel (y-coordinate)
                if use_loss[1]:
                    x_coord -= tensor[i, j, 3].item()
                    y_coord -= tensor[i, j, 4].item()
                
                # Add the detection info to the dictionary
                if data is None:
                    detected_circles[counter] = {
                        's': score,
                        'x': x_coord,
                        'y': y_coord,
                        'r': radius_detection_circle
                    }
                    # Increment the counter for the next detection
                    counter += 1
                else:
                    detected_circles.append({"id": id_counter,
                                  "keypoints": [[0,0,0], [x_coord, y_coord, 0]],
                                  "position_on_pitch": [x_coord, y_coord, 0],
                                  "score": score,
                                  "image_id": int(data[1]),
                                  "category_id": 1,
                                  "area": 0,
                                  'x': x_coord,
                                  'y': y_coord,
                                  "r": radius_detection_circle})
                    id_counter += 1

                
    if data is None:
        return detected_circles
    return detected_circles, id_counter

def run_model(params, debug=False, save_for_nms=True, quantisation="static", test=False):
    if quantisation not in ["half", "static", "none"]:
        raise ValueError("Quantisation must be one of 'static', 'half', or 'none'")
    if debug:
        p = "mini"
    else:
        if test:
            p = "test"
        else:
            p = "val"
    with open(f'{params[0]}annotations/{p}.json', 'r') as file:
        data_val = json.load(file)
    with open(f'{params[0]}annotations/train.json', 'r') as file:
        data_train = json.load(file)
    val_images = data_val.get("images", [])
    trn_images = data_train.get("images", [])
    val_annotations = data_val.get("annotations", [])
    trn_annotations = data_train.get("annotations", [])
    val_images_dict = {image["id"]: image for image in val_images}
    trn_images_dict = {image["id"]: image for image in trn_images}
    val_annotations_dict = {}
    for annotation in val_annotations:
        image_id = annotation["image_id"]
        if image_id not in val_annotations_dict:
            val_annotations_dict[image_id] = []
        val_annotations_dict[image_id].append(annotation)
    val_dataset = BEVDataset(val_images_dict, image_dir=f'{params[0]}{p}/')
    trn_dataset = BEVDataset(trn_images_dict, image_dir=f'{params[0]}train/')
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = params[1]
    res_net_tail = params[3]
    layer_params = params[2]
    resblock_channels = params[6]
    sampling_strategy = params[7]
    if quantisation == "QAT":
        backend = "qnnpack"          # keep same backend used during QAT
        torch.backends.quantized.engine = backend
    if backbone == "unet" or backbone == "mobile-unet":
        model = BEVNetwork(annotations=trn_annotations, feature_extractor=backbone,
                           feature_channels=layer_params[-1], unet_features=layer_params[:-1], 
                           res_net_tail=res_net_tail, res_block_channels=resblock_channels,
                           tail_param=resblock_channels, sampling_strategy=sampling_strategy)
    else:
        model = BEVNetwork(annotations=trn_annotations, feature_channels=256,
                           feature_extractor=backbone, res_net_tail=res_net_tail)

    model.load_state_dict(torch.load(f"./models/{params[4]}.pt", map_location="cpu", weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weight = torch.as_tensor(np.load("bce_weight.npy"))
    loss_fkt = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    sig = torch.nn.Sigmoid()

    val_epoch_loss = 0.0
    validation_pos_est = []
    id_counter = 0
    model.to(device)

    if save_for_nms:
        output_dir = f"outputs/{params[4]}-{quantisation}-{p}"
        os.makedirs(output_dir, exist_ok=True)
        print("Saving outputs to:", output_dir)
    # skip if outputs already present
    existing = [f for f in os.listdir(output_dir) if f.endswith(".pth") and os.path.isfile(os.path.join(output_dir, f))]
    count = len(existing)
    expected = len(val_data_loader) * getattr(val_data_loader, "batch_size", 1)
    if count >= expected:
        print(f"Found {count} .pth files in {output_dir} (expected {expected}), skipping.")
        return

    if quantisation == "static":
        device = torch.device("cpu")
        model.to(device)
        model_orig = copy.deepcopy(model)
        tmp_images_dict = {k: v for k, v in val_images_dict.items() if k in range(200)}
        tmp_dataset = BEVDataset(tmp_images_dict, image_dir=f'{params[0]}{p}/')
        tmp_data_loader = DataLoader(tmp_dataset, batch_size=2, shuffle=False, num_workers=0)
        torch.backends.quantized.engine = "qnnpack"
        model = model.eval().to(device)
        model.qconfig = get_default_qconfig(backend="qnnpack") #fbgemm
        #fuse_model(model)
        #wrap_convtranspose_as_float_island(model)
        wrap_named_module_as_float_island(model, "project_and_sample_voxel_grid", model.qconfig)
        model = prepare(model, inplace=False)
        with torch.no_grad():
            for idx, (image, dist_poly, camera_matrix, _, _) in enumerate(tmp_data_loader):
                image = image.to(device)
                dist_poly = dist_poly.to(device)
                camera_matrix = camera_matrix.to(device)
                _ = model(image, dist_poly, camera_matrix)
        convert(model, inplace=True)
    elif quantisation == "half":
        model_orig = copy.deepcopy(model)
        model.half()
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.float()
    if debug:
        model.to(device)
        if quantisation == "half":
            input_im = torch.randn(1, 3, 3840, 2160).half().to(device)
            input_poly = torch.randn(1, 9).half().to(device)
            input_cm = torch.randn(1, 3, 4).half().to(device)
        else:
            input_im = torch.randn(1, 3, 3840, 2160).to(device)
            input_poly = torch.randn(1, 9).to(device)
            input_cm = torch.randn(1, 3, 4).to(device)
        macs, params = profile(model, inputs=(input_im, input_poly, input_cm))
        print(f"GFLOPs: {macs * 2 / 1e9}, Params: {params}")

    if model.feat_extract.final_conv.out_channels >= 32:
        device = torch.device("cpu")
        model.to(device)
    for batch_idx, (image_vals, dist_poly, camera_matrix, im_idxs, file_name) in enumerate(val_data_loader):
        if save_for_nms:
            output_path = os.path.join(output_dir, f"{file_name[-1]}.pth")
            if os.path.exists(output_path):
                continue
        if quantisation == "half":
            image_vals = image_vals.half()
            dist_poly = dist_poly.half()
            camera_matrix = camera_matrix.half()
        with torch.no_grad():
            output, gt_io, e_loss, val_dict, id = run_batch(False, device, optimizer, model, image_vals, dist_poly, camera_matrix, im_idxs, 
                 loss_fkt, val_annotations_dict, sig, file_name, id_counter=id_counter, params=params[5])
        if save_for_nms:
            for idx, fname in enumerate(file_name):
                center = - torch.tensor([model.voxel_grid_size_col/2, model.voxel_grid_size_row/2])
                gnd = grid2d(model.voxel_grid_size_col * model.voxel_res, 
                    model.voxel_grid_size_row * model.voxel_res) / model.voxel_res + center
                # Col_max and row_max are on the far side to the left (in the image)
                gnd = torch.flip(gnd, (0, 1, 2))
                est_d = (sig(output.permute(0,2,3,1)[:,:,:,1:]) - 0.5) * 2/model.voxel_res
                tensor = sig(output[idx, 0:1, :, :]).permute(1, 2, 0)
                if debug:
                    # Normalize tensor to [0, 255] for visualization
                    img = tensor[..., 0].detach().cpu().numpy().copy()
                    img *= 255
                    img = img.astype(np.uint8)
                    img_path = os.path.join(output_dir, f"{fname}_tensor.png")
                    cv2.imwrite(img_path, img)
                tensor = torch.cat((tensor, gnd.to(tensor.device), est_d[0,:,:,:]), dim=-1)
                data=[id_counter, int(fname), [True, True]]
                detected_circles, id_counter = create_detections_from_tensor(tensor, 0.3, 1, data)
                torch.save(detected_circles, os.path.join(output_dir, f"{fname}.pth"))
                if debug:
                    present_image_and_bev(save=True, qual_eval=True, score_threshold=0.95,
                                          est_data_dict=detected_circles, test=True)
        val_epoch_loss += sum(e_loss)
        validation_pos_est.extend(val_dict)
        id_counter = id

def present_image_and_bev(debug=False, save_dir:str="./qualitative_results/", disp_est:str=None, 
                          data_dir="./data/", score_threshold=0.5, im_stop=10):
    _, _, image_dict, annotations_dict = load_annotations(debug, data_dir, get_val=True)

    for id_im, _ in enumerate(image_dict):
        image_data = image_dict[id_im]
        if id_im > im_stop:
            break
        file_name = image_data['file_name']
        if debug:
            image_path = f'{data_dir}mini/{file_name}'
        else:
            image_path = f'{data_dir}train/{file_name}'
        image = cv2.imread(image_path)

        bev_tensor = project_on_ground(image_data['camera_matrix'], image_data['dist_poly'],
                                       imread(image_path)[None] + 1e-6, 110, 110, 10)[0]
        bev_view = bev_tensor.numpy().transpose(1,2,0)[:,:,::-1].copy() * 255
        msk = bev_view.sum(2) == 0
        for i in range(3):
            bev_view[:,:,i][msk] = 255

        for annotation in annotations_dict[id_im]:
            # Plotting ground truth annotations
            x_min, y_min, width, height = annotation["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            pos_on_pitch = ((np.array(annotation['position_on_pitch']) * 10)).astype(np.int16)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), [255,0,0], 3)
            cv2.circle(bev_view, (pos_on_pitch[0]+int(bev_view.shape[0]/2), 
                                  pos_on_pitch[1]+int(bev_view.shape[1]/2)), 8, [255, 0, 0], -1)

        if disp_est is not None:
            with open(disp_est, 'r') as file:
                est_data = json.load(file)

            # Filter annotations based on scores
            filtered_annotations = [anno for anno in est_data['annotations'] if anno['score'] > score_threshold]
            for annotation in filtered_annotations:
                if annotation["image_id"] != id_im:
                    continue
                image_id = annotation["image_id"]
                if image_id not in annotations_dict:
                    annotations_dict[image_id] = []
                annotations_dict[image_id].append(annotation)
                x_max = x_min + width
                y_max = y_min + height
                pos_on_pitch = ((np.array(annotation['position_on_pitch'][:2]) * 10)).astype(np.int16)
                cv2.circle(bev_view, (pos_on_pitch[0]+int(bev_view.shape[0]/2),
                                      pos_on_pitch[1]+int(bev_view.shape[1]/2)), 10, [0, 0, 255], 2)
        bev_view = cv2.rotate(bev_view, cv2.ROTATE_90_CLOCKWISE)
        bev_view = cv2.flip(bev_view, 0)

        bev_height, bev_width, _ = bev_view.shape
        img_height, _ = image.shape[:2]
        aspect_ratio = bev_width / bev_height
        new_bev_width = img_height * aspect_ratio
        bev_view_resized = cv2.resize(bev_view, (int(new_bev_width), img_height),
                                      interpolation=cv2.INTER_AREA)
        bev_view_resized = np.clip(bev_view_resized, 0, 255).astype(np.uint8)
        concatenated_image = np.concatenate((image, bev_view_resized), axis=1)
        scale_percent = 30
        w = int(concatenated_image.shape[1] * scale_percent / 100)
        h = int(concatenated_image.shape[0] * scale_percent / 100)
        concatenated_image = cv2.resize(concatenated_image, (w, h), interpolation = cv2.INTER_AREA)

        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}bev-{id_im}.png", bev_view)
        cv2.imwrite(f"{save_dir}{id_im}.png", concatenated_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_mode", help = "Runs code in debug mode",
                       action="store_true")
    parser.add_argument("-p", "--data_path", help = "Path to directory where data is found")
    parser.add_argument("-s", "--save_path", help = "Path to directory where data is saved")
    parser.add_argument("-f", "--feat_extract", help = "Name of feature extractor")
    parser.add_argument("-pm", "--parameters", help = "Parameters for the model")
    parser.add_argument("-r", "--res_net_tail", help = "ResNet tail parameters")
    parser.add_argument("-mn", "--model_name", help = "Model name to load")
    parser.add_argument("-xy", "--xy_corr", help = "Remove xy correction from model",
                       action="store_false")
    parser.add_argument("-rp", "--res_net_parameters", help = "ResNet parameters")
    parser.add_argument("-ss", "--sampling_strategy", help = "Sampling strategy to use")
    parser.add_argument("-q", "--quantisation", help = "Quantisation method to use")
    args = parser.parse_args()
    debug = args.debug_mode
    data_path = args.data_path
    save_dir = args.save_path
    feat_extract = args.feat_extract
    params = args.parameters
    if params is not None:
        params = params.split(",")
        params = [int(p) for p in params]
    res_net_tail = args.res_net_tail
    if res_net_tail not in ["resnet", "3dConv", "2dConv", "none"]:
        raise ValueError("ResNet tail must be one of 'resnet', '3dConv', '2dConv', or 'none'")
    res_net_parameters = args.res_net_parameters
    model_name = args.model_name
    xy_corr = args.xy_corr
    sampling_strategy = args.sampling_strategy
    quantisation = args.quantisation
    model_params = [data_path, feat_extract, params, res_net_tail, model_name, xy_corr, 
                    res_net_parameters, sampling_strategy]
    run_model(model_params, debug, save_for_nms=True, quantisation=quantisation, test=False)