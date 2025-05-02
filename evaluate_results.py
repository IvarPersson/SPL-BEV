import argparse
import json
import os

import cv2
import numpy as np
from sskit import project_on_ground, imread

from utils import load_annotations

def present_image_and_bev(debug=False, save_dir:str="./qualitative_results/", disp_est:str=None, data_dir="./data/", score_threshold=0.5, im_stop=10):
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

        bev_tensor = project_on_ground(image_data['camera_matrix'], image_data['dist_poly'], imread(image_path)[None] + 1e-6, 110, 110, 10)[0]
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
            cv2.circle(bev_view, (pos_on_pitch[0]+int(bev_view.shape[0]/2), pos_on_pitch[1]+int(bev_view.shape[1]/2)), 8, [255, 0, 0], -1)

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
                cv2.circle(bev_view, (pos_on_pitch[0]+int(bev_view.shape[0]/2), pos_on_pitch[1]+int(bev_view.shape[1]/2)), 10, [0, 0, 255], 2)
        bev_view = cv2.rotate(bev_view, cv2.ROTATE_90_CLOCKWISE)
        bev_view = cv2.flip(bev_view, 0)

        bev_height, bev_width, _ = bev_view.shape
        img_height, _ = image.shape[:2]
        aspect_ratio = bev_width / bev_height
        new_bev_width = img_height * aspect_ratio
        bev_view_resized = cv2.resize(bev_view, (int(new_bev_width), img_height), interpolation=cv2.INTER_AREA)
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
    args = parser.parse_args()
    debug = args.debug_mode
    data_path = args.data_path
    save_dir = args.save_path
    present_image_and_bev(debug, save_dir=save_dir, disp_est=None, data_dir=data_path, score_threshold=0.76)