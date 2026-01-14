import os

import torch
from matplotlib import pyplot as plt

from utils import create_ground_truth_tensor, create_nms_output

def run_batch(debug, train, device, optimizer, model, image_vals, dist_poly, camera_matrix, 
              im_idxs, criterion, annotations_dict, sig, file_name, running_loss=0, id_counter=0, 
              params=None):
    if params is not None:
        use_loss = [True, params]
    else:
        use_loss = [True, True]
    optimizer.zero_grad()
    output = model(image_vals.to(device), dist_poly.to(device), camera_matrix.to(device))
    gt_io, gt_dist, gnd = create_ground_truth_tensor(model, output, annotations_dict, im_idxs)

    # Calculate loss
    loss1 = criterion(output[:,0,:,:], gt_io)
    est_d = (sig(output.permute(0,2,3,1)[:,:,:,1:]) - 0.5) * 2/model.voxel_res
    tmp_o = est_d[torch.where(gt_dist != 1e4)].reshape(-1,2)
    tmp_gt = gt_dist[torch.where(gt_dist != 1e4)].reshape(-1,2)
    loss2 = torch.norm(tmp_o - tmp_gt, dim=1).mean()
    loss = 0
    if use_loss[0]:
        loss += 0.1*loss1
    if use_loss[1]:
        loss += loss2
    if train:
        loss.backward()
        optimizer.step()
        running_loss += (loss1.item() + loss2.item())
    batch_est = []
    if debug and not train:
        batch_est, id_counter = create_nms_output(model, output, est_d, sig, im_idxs,
                                                  use_loss, id_counter, (1,0.5,0))
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        output_io = sig(output[0, 0, :, :]).detach().cpu().numpy()
        plt.imshow(output_io, cmap='Greys', interpolation='nearest')
        plt.title(f"Model output IO heatmap on image {file_name[0]}")
            
        plt.subplot(2, 2, 2)
        cutoff = 0.5
        bev_cutoff = sig(output[0, 0, :, :]).detach().cpu().numpy() > cutoff
        plt.imshow(bev_cutoff, cmap='Greys', interpolation='nearest')
        plt.title(f"Estimated BEV, cutoff: {cutoff}")

        plt.subplot(2, 2, 3)
        gt_io_view = gt_io[0].detach().cpu().numpy()
        plt.imshow(gt_io_view, cmap='Greys', interpolation='nearest')
        plt.title(f"Ground truth IO heatmap on image {file_name[0]}")

        plt.subplot(2, 2, 4)
        id = batch_est[0]["image_id"]
        for be in batch_est:
            if be["image_id"] == id:
                plt.plot(-be["position_on_pitch"][1], be["position_on_pitch"][0], 'ro')
        plt.title(f"Ground truth IO heatmap on image {file_name[0]}")
        
        save_dir = "./train_over_time/"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}{file_name[0]}_e_{0}_output_vs_gt.png")
        plt.close()

    losses = [loss1.item()*use_loss[0], loss2.item()*use_loss[1]]
    return output, gt_io, losses, batch_est, id_counter

