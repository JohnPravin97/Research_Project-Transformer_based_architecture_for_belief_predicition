import pandas as pd
import numpy as np
import pickle as pkl
import cv2
import os
import ast
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from natsort import natsorted
from torch.nn.utils.rnn import pad_sequence
from configs.param import TRAIN_FRAME_INPUT_PATH, PREPROCESSED_TRAIN_FRAME_INPUT_PATH, RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT
from configs.param import POSE_AS_INPUT, BBOX_AS_INPUT, USE_OCR_AS, TRAIN_POSE_INPUT_PATH
from prettytable import PrettyTable
import logging

def summary_table_plotter(model, load_model_path):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    if (BBOX_AS_INPUT == "new_coordinates" or BBOX_AS_INPUT == "old_coordinates") and (POSE_AS_INPUT == "coordinates"):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            if name.startswith("bbox_layers"): continue
            if "pose_graph" in name: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        logging.info(table)
        logging.info("Total Trainable Params: {}".format(total_params))

    elif (BBOX_AS_INPUT == "picture") and (POSE_AS_INPUT == "coordinates"):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            if name.startswith("bbox_ff"): continue
            if "pose_graph" in name: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        logging.info(table)
        logging.info("Total Trainable Params: {}".format(total_params))
    
    elif (BBOX_AS_INPUT == "new_coordinates" or BBOX_AS_INPUT == "old_coordinates") and (POSE_AS_INPUT == "graph"):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            if name.startswith("bbox_layers"): continue
            if "pose_ff" in name: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        logging.info(table)
        logging.info("Total Trainable Params: {}".format(total_params))
    
    elif (BBOX_AS_INPUT == "new_coordinates" or BBOX_AS_INPUT == "old_coordinates") and (POSE_AS_INPUT == "picture"):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            if name.startswith("bbox_layers"): continue
            if "pose_ff" in name: continue
            if "pose_graph" in name: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        logging.info(table)
        logging.info("Total Trainable Params: {}".format(total_params))
    
    elif (BBOX_AS_INPUT == "picture") and (POSE_AS_INPUT == "graph"):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            if name.startswith("bbox_ff"): continue
            if "pose_ff" in name: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        logging.info(table)
        logging.info("Total Trainable Params: {}".format(total_params))
    
    with open(os.path.join(load_model_path, 'model_summary.txt'), 'w') as f:
        f.write('{}\n'.format(table))
        f.write("Total Trainable Params: {}".format(total_params))
        f.close()

    return None

class CustomLoss(nn.Module):
    def __init__(self, ocr_matrix):
        super(CustomLoss, self).__init__()
        self.ocr_matrix = ocr_matrix
        self.loss = nn.L1Loss(reduction='mean')
        
    def forward(self, true_left_label, true_right_label, predicted_label):
        true_label = torch.zeros_like(predicted_label)
        for i in range(true_label.shape[0]):
            true_label[i] = self.ocr_matrix[true_left_label[i]][true_right_label[i]]
        #logging.info("Predicted ocr {}".format(predicted_label))
        #logging.info("True ocr {}".format(true_label))
        out = self.loss(true_label, predicted_label)
        return out

def presave(save_folder):
    preprocess = transforms.Compose([
        transforms.Resize((RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT)),
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_path = TRAIN_FRAME_INPUT_PATH
    frame_dirs = os.listdir(frame_path)
    frame_paths = []
    for frame_dir in natsorted(frame_dirs):
        #print(frame_dir)
        paths = [os.path.join(frame_path, frame_dir, i) for i in natsorted(os.listdir(os.path.join(frame_path, frame_dir)))]
        frame_paths.append(paths)

    #save_folder = os.path.join(PREPROCESSED_TRAIN_FRAME_INPUT_PATH, frame_path.split('/')[-2], frame_path.split('/')[-1])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(save_folder, 'created')

    for video in natsorted(frame_paths):
        #print(video[0].split('/')[-2], end='\r')
        images = [preprocess(Image.open(i)) for i in video]
        strings = video[0].split('/')
        with open(save_folder+'/'+strings[7]+'.pkl', 'wb') as f:
            pkl.dump(images, f)

def presave_bbox_func(new_bbox_path, resized_bbox_path, episode_ids):
    bbox_complete = []
    if new_bbox_path is not None:
        new_bbox_path = os.path.join(new_bbox_path, 'crops')
        resized_bbox_path = os.path.join(resized_bbox_path, 'crops')
        for episode_id in episode_ids:
            bbox_dir = os.path.join(new_bbox_path, episode_id)
            resized_dir = os.path.join(resized_bbox_path, episode_id)
            if not os.path.exists(resized_dir):
                os.makedirs(resized_dir)
            bbox_frame_tensor = []
            for bbox_per_frame in natsorted(os.listdir(bbox_dir)):
                bbox_frame_dir = os.path.join(bbox_dir, bbox_per_frame)
                resized_bbox_frame_dir = os.path.join(resized_dir, bbox_per_frame)
                if not os.path.exists(resized_bbox_frame_dir):
                    os.makedirs(resized_bbox_frame_dir)
                for bbox_objects in natsorted(os.listdir(bbox_frame_dir)):
                    image = io.imread(os.path.join(bbox_frame_dir, bbox_objects))
                    ind_bbox_object = resize(image,(image.shape[0] // 4, image.shape[1] // 4),  anti_aliasing=True) 
                    io.imsave(os.path.join(resized_bbox_frame_dir, bbox_objects), img_as_ubyte(ind_bbox_object))
    return None

def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    xv, yv = torch.meshgrid(torch.tensor(range(img_width)), torch.tensor(range(img_height)), indexing='xy')

    a = torch.cos(theta) ** 2 / (2 * var_x) + torch.sin(theta) ** 2 / (2 * var_y)
    b = -torch.sin(2 * theta) / (4 * var_x) + torch.sin(2 * theta) / (4 * var_y)
    c = torch.sin(theta) ** 2 / (2 * var_x) + torch.cos(theta) ** 2 / (2 * var_y)

    return torch.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))

def poses_pictures_make_func(joints, limbs, sigma_parallel, sigma_perp, reshape_width, reshape_height):
    mask = torch.zeros((reshape_width, reshape_height, len(limbs)))
    for i in range(len(limbs)):
        n_joints_for_limb = len(limbs[i])
        p = torch.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p[j, :] = torch.tensor([joints[limbs[i][j], 0], joints[limbs[i][j], 1]])

        center = torch.mean(p, 0)

        #sigma_parallel = torch.max(torch.tensor([5, (torch.sum((p[1, :] - p[0, :]) ** 2)) / 1.5]))
        theta = torch.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

        mask_i = make_gaussian_map(reshape_width, reshape_height, center, sigma_parallel[i], sigma_perp[i], theta)
        mask[:, :, i] = mask_i / (torch.amax(mask_i) + 1e-6)

    src_bg_mask = 1.0 - torch.amax(mask, axis=2)
    src_bg_mask = src_bg_mask[:,:, None]
    final_src_masks = torch.log(torch.cat((src_bg_mask, mask), axis=2) + torch.tensor(1e-10))
    return final_src_masks

def presave_pose_as_picture(save_folder): #define save folder path to save the pose as picture pkl files
    sigma_perp = np.array([threshold for i in range(len(POSE_PICTURE_LIMBS))])
    sigma_parallel = np.array([threshold for i in range(len(POSE_PICTURE_LIMBS))])
    NO_NEED_POSE_JOINTS = [11, 14, 19, 20, 21, 22, 23, 24]
    threshold = 0.1
    pose_left_complete, pose_right_complete, episode_complete = [], [], []
    pose_path = TRAIN_POSE_INPUT_PATH
    pose_dirs = os.listdir(pose_path)
    pose_paths = []
    for episode_id in natsorted(pose_dirs):
        if episode_id not in ["0", "1"]:
            print(episode_id)
            pose_dir = os.path.join(pose_path, episode_id)
            pose_list = []
            poses = natsorted(os.listdir(pose_dir))
            final_left_mask = torch.zeros((len(poses), RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT, 17))
            final_right_mask = torch.zeros((len(poses), RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT, 17))

            final_left_pose = torch.zeros((len(poses), 1, 17, 3))
            final_right_pose = torch.zeros((len(poses), 1, 17, 3))
            for i, pose in enumerate(poses):
                fs = cv2.FileStorage(os.path.join(pose_dir, pose), cv2.FILE_STORAGE_READ)
                if torch.tensor(fs.getNode("pose_0").mat()).shape != (2,25,3):
                    pose_list.append(fs.getNode("pose_0").mat()[:2,:,:])
                else:
                    pose_list.append(fs.getNode("pose_0").mat())

                # To remove the no need limbs from the pose data
                final_left_pose[i][0] = torch.tensor(np.delete(pose_list[0][0], NO_NEED_POSE_JOINTS, axis=0))
                final_right_pose[i][0] = torch.tensor(np.delete(pose_list[0][1], NO_NEED_POSE_JOINTS, axis=0))

                left_joints = torch.zeros((17, 2))
                right_joints = torch.zeros((17, 2))

                left_joints[:,0] = (final_left_pose[i][0][:, 0] / ORIGINAL_IMG_HEIGHT) * RESHAPED_IMG_HEIGHT
                left_joints[:,1] = (final_left_pose[i][0][:, 1]/ ORIGINAL_IMG_WIDTH) * RESHAPED_IMG_WIDTH

                right_joints[:,0] = (final_right_pose[i][0][:, 0]/ ORIGINAL_IMG_HEIGHT) * RESHAPED_IMG_HEIGHT
                right_joints[:,1] = (final_right_pose[i][0][:, 1]/ ORIGINAL_IMG_WIDTH) * RESHAPED_IMG_WIDTH

                final_left_mask[i] = poses_pictures_make_func(left_joints, POSE_PICTURE_LIMBS, sigma_parallel, sigma_perp, RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT)
                #print(final_left_mask[i].shape)

                final_right_mask[i] = poses_pictures_make_func(right_joints, POSE_PICTURE_LIMBS, sigma_parallel, sigma_perp, RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT)

            with open(save_folder+'\\'+"left_"+episode_id+'.pkl', 'wb') as left_f:
                pkl.dump(final_left_mask, left_f)
            left_f.close()
            print("Done with left pose -- created")

            with open(save_folder+'\\'+"right_"+episode_id+'.pkl', 'wb') as right_f:
                pkl.dump(final_right_mask, right_f)
            right_f.close()
            print("Done with right pose -- created")
        else:
            continue


if __name__=='__main__':
    assert("This can not be run as a single file")