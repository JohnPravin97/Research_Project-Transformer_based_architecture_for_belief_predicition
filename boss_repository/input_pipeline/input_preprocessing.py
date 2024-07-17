import os
import pickle
import cv2
import torch
import ast
import numpy as np
from skimage import io
import logging
from natsort import natsorted
import pickle as pkl
from configs.param import OBJECTS, NO_NEED_POSE_JOINTS, ORIGINAL_IMG_WIDTH, ORIGINAL_IMG_HEIGHT, RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT, OCR_GRAPH, POSE_PICTURE_LIMBS
from utils.utils_func import poses_pictures_make_func
from torchvision import transforms

def frame_func(frame_path, frame_dirs):
    frame_paths_complete = []
    sequence_length_complete = []
    for frame_id in natsorted(frame_dirs):
        with open(os.path.join(frame_path, frame_id) , "rb") as f:
            frame_paths = pkl.load(f)
            sequence_length_complete.append(len(frame_paths))
            frame_paths_complete.append(frame_paths)  

    logging.info("Frame path length is {}".format(len(frame_paths_complete)))
    return frame_paths_complete, sequence_length_complete


def label_func(label_path, episode_ids):
    labels_complete = []
    with open(label_path, 'rb') as fp:
        labels = pickle.load(fp) #labels consists of all 300 experiments labels
        left_labels, right_labels = labels
        #below loop is used to take the training samples labels out of the complete train-val-test labels
        for episode_id in episode_ids:
            left_labels_i = left_labels[int(episode_id)]
            right_labels_i = right_labels[int(episode_id)]
            labels_complete.append([left_labels_i, right_labels_i])
        fp.close()

    logging.info("Label length is {} ".format(len(labels_complete)))
    return labels_complete

def pose_func(pose_path, episode_ids, pose_as_input, pose_as_picture_path):
    pose_left_complete, pose_right_complete = [], []

    if pose_as_input == "coordinates":
        ## Note
        if pose_path is not None:
            for episode_id in episode_ids:  
                pose_dir = os.path.join(pose_path, episode_id)
                pose_list = []
                pose = natsorted(os.listdir(pose_dir))
                final_left_pose = torch.zeros((len(pose), 1, 17, 3))
                final_right_pose = torch.zeros((len(pose), 1, 17, 3))
                for i, pose in enumerate(pose):
                    fs = cv2.FileStorage(os.path.join(pose_dir, pose), cv2.FILE_STORAGE_READ)
                    if torch.tensor(fs.getNode("pose_0").mat()).shape != (2,25,3):
                        pose_list.append(fs.getNode("pose_0").mat()[:2,:,:])
                    else:
                        pose_list.append(fs.getNode("pose_0").mat())
                        
                    # To remove the no need limbs from the pose data
                    final_left_pose[i][0] = torch.tensor(np.delete(pose_list[i][0], NO_NEED_POSE_JOINTS, axis=0))
                    final_right_pose[i][0] = torch.tensor(np.delete(pose_list[i][1], NO_NEED_POSE_JOINTS, axis=0))
                    # To normalize the pose value
                    final_left_pose[i][0][:, 0] = final_left_pose[i][0][:, 0]/ORIGINAL_IMG_WIDTH
                    final_left_pose[i][0][:, 1]= final_left_pose[i][0][:, 1]/ORIGINAL_IMG_HEIGHT

                    final_right_pose[i][0][:, 0]= final_right_pose[i][0][:, 0]/ORIGINAL_IMG_WIDTH
                    final_right_pose[i][0][:, 1]= final_right_pose[i][0][:, 1]/ORIGINAL_IMG_HEIGHT

                    #final_pose.shape = (2,17,3)
                pose_left_complete.append(torch.flatten(final_left_pose, 1))
                pose_right_complete.append(torch.flatten(final_right_pose, 1))
                 

    elif pose_as_input == "graph":
        ## Note
        if pose_path is not None:
            for episode_id in episode_ids:  
                pose_dir = os.path.join(pose_path, episode_id)
                pose_list = []
                pose = natsorted(os.listdir(pose_dir))
                final_left_pose = torch.zeros((len(pose), 17, 3))
                final_right_pose = torch.zeros((len(pose), 17, 3))
                for i, pose in enumerate(pose):
                    fs = cv2.FileStorage(os.path.join(pose_dir, pose), cv2.FILE_STORAGE_READ)
                    if torch.tensor(fs.getNode("pose_0").mat()).shape != (2,25,3):
                        pose_list.append(fs.getNode("pose_0").mat()[:2,:,:])
                    else:
                        pose_list.append(fs.getNode("pose_0").mat())

                    # To remove the no need limbs from the pose data
                    final_left_pose[i]= torch.tensor(np.delete(pose_list[i][0], NO_NEED_POSE_JOINTS, axis=0))
                    final_right_pose[i] = torch.tensor(np.delete(pose_list[i][1], NO_NEED_POSE_JOINTS, axis=0))
                    # To normalize the pose value
                    final_left_pose[i][:, 0] = final_left_pose[i][:, 0]/ORIGINAL_IMG_WIDTH
                    final_left_pose[i][:, 1]= final_left_pose[i][:, 1]/ORIGINAL_IMG_HEIGHT

                    final_right_pose[i][:, 0]= final_right_pose[i][:, 0]/ORIGINAL_IMG_WIDTH
                    final_right_pose[i][:, 1]= final_right_pose[i][:, 1]/ORIGINAL_IMG_HEIGHT

                #final_pose.shape = (2,17,3)
                pose_left_complete.append(final_left_pose) 
                pose_right_complete.append(final_right_pose)


    elif pose_as_input == "picture": 
        if pose_as_picture_path is not None:
            pose_list = []
            left_pose_path = os.path.join(pose_as_picture_path, "left_pose")
            right_pose_path = os.path.join(pose_as_picture_path, "right_pose")
            left_pose_dir = natsorted(os.listdir(left_pose_path))
            right_pose_dir = natsorted(os.listdir(right_pose_path))
            for pose_id in natsorted(left_pose_dir):
                with open(os.path.join(left_pose_path, pose_id) , "rb") as f:
                    left_pose_paths = pkl.load(f)
                    pose_left_complete.append(left_pose_paths)  
            for pose_id in natsorted(right_pose_dir):
                with open(os.path.join(right_pose_path, pose_id) , "rb") as f:
                    right_pose_paths = pkl.load(f)
                    pose_right_complete.append(right_pose_paths)  
    else:
        logging.info("please choose between ['picture', 'graph', 'coordinates']")

    # poses.shape = [2, 25, 3]
    logging.info("poses left length is {}".format(len(pose_left_complete)))
    logging.info("poses right length is {}".format(len(pose_right_complete)))
    return pose_left_complete, pose_right_complete

def gaze_func(gaze_path, episode_ids):
    gaze_left_complete, gaze_right_complete = [], []
    if gaze_path is not None:
        for episode_id in episode_ids:
            gaze_txt = os.path.join(gaze_path, '{}.txt'.format(episode_id))
            with open(gaze_txt, 'r') as fp:
                gaze_content = fp.readlines()
                fp.close()
            gaze_content = ast.literal_eval(gaze_content[0])
            gaze_left_tensor = torch.zeros((len(gaze_content), 3))
            gaze_right_tensor = torch.zeros((len(gaze_content), 3))
            for j in range(len(gaze_content)):
                if len(gaze_content[j]) >= 2:
                    gaze_left_tensor[j,:] = torch.tensor(gaze_content[j][:2][0])
                    gaze_right_tensor[j,:] = torch.tensor(gaze_content[j][:2][1])
                elif len(gaze_content[j]) == 1:
                    gaze_left_tensor[j,0] = torch.tensor(gaze_content[j][0][0])
                    gaze_right_tensor[j,0] = torch.tensor(gaze_content[j][0][1])
                else:
                    continue
        
            gaze_left_complete.append(gaze_left_tensor)
            gaze_right_complete.append(gaze_right_tensor)

    #gaze_i_tensor.shape(total_frame, 2, 3)
    logging.info("gaze left length is {}".format(len(gaze_left_complete)))
    logging.info("gaze right length is {}".format(len(gaze_right_complete)))
    return gaze_left_complete, gaze_right_complete

def bbox_func(new_bbox_path, old_bbox_path, episode_ids, objects, bbox_as_input):
    bbox_complete = []
    #episode_ids = ["15"]
    if bbox_as_input == "new_coordinates":
        if new_bbox_path is not None:
            new_bbox_path = os.path.join(new_bbox_path, 'labels')
            for episode_id in episode_ids:
                bbox_dir = os.path.join(new_bbox_path, episode_id)
                bbox_tensor = torch.zeros((len(os.listdir(bbox_dir)), len(objects), 4))
                for index, bbox in enumerate(natsorted(os.listdir(bbox_dir))):
                    with open(os.path.join(bbox_dir, bbox), 'r') as fp:
                        bbox_content = fp.readlines()
                        fp.close()
                    for bbox_content_line in bbox_content:
                        bbox_content_values = bbox_content_line.split()
                        class_index, x_center, y_center, x_width, y_height = map(float, bbox_content_values)
                        bbox_tensor[index][int(class_index)] = torch.FloatTensor([x_center, y_center, x_width, y_height])
                
                bbox_complete.append(torch.flatten(bbox_tensor, 1))
                #bbox_complete.append(bbox_tensor)
    
    elif bbox_as_input == "old_coordinates":
        if old_bbox_path is not None:
            #old_bbox_path = os.path.join(old_bbox_path, 'labels')
            for episode_id in episode_ids:
                bbox_dir = os.path.join(old_bbox_path, episode_id)
                with open(bbox_dir, 'rb') as fp:
                    bboxes_i = pickle.load(fp)
                    len_i = len(bboxes_i)
                    fp.close()
                bboxes_i_tensor = torch.zeros((len_i, len(objects), 4))
                for j in range(len(bboxes_i)):
                    items_i_j, bboxes_i_j = bboxes_i[j]
                    for k in range(len(items_i_j)):
                        bboxes_i_tensor[j, objects.index(items_i_j[k])] = torch.tensor([
                                                                                bboxes_i_j[k][0] / 1920 ,
                                                                                bboxes_i_j[k][1] / 1088 ,
                                                                                bboxes_i_j[k][2] / 1920 ,
                                                                                bboxes_i_j[k][3] / 1088
                                                                            ])

                #print(torch.flatten(bboxes_i_tensor, 1))        
                bbox_complete.append(torch.flatten(bboxes_i_tensor, 1))
                #bbox_complete.append(bbox_tensor)

    elif bbox_as_input=="picture":
        if new_bbox_path is not None:
            preprocess = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            new_bbox_path = os.path.join(new_bbox_path, 'crops')
            for episode_id in episode_ids:
                bbox_dir = os.path.join(new_bbox_path, episode_id)
                bbox_frame_tensor = []
                for bbox_per_frame in natsorted(os.listdir(bbox_dir)):
                    bbox_frame_dir = os.path.join(bbox_dir, bbox_per_frame)
                    bbox_objects_tensor = [preprocess(io.imread(os.path.join(bbox_frame_dir, bbox_objects))) for bbox_objects in natsorted(os.listdir(bbox_frame_dir))]
                    # fake_object_tensor = torch.zeros((32, 32, 3))
                    fake_object_tensor = torch.zeros((3, 8, 8))
                    if len(bbox_objects_tensor) < 10:
                        for i in range(len(bbox_objects_tensor), 10):
                            bbox_objects_tensor.append(fake_object_tensor)
                        bbox_frame_tensor.append(torch.stack(bbox_objects_tensor))
                    elif len(bbox_objects_tensor) > 10:
                        bbox_frame_tensor.append(torch.stack(bbox_objects_tensor[0:10]))
                    elif len(bbox_objects_tensor) == 10:
                        bbox_frame_tensor.append(torch.stack(bbox_objects_tensor))
                bbox_complete.append(torch.stack(bbox_frame_tensor))

    else:
        logging.info("please choose between ['picture', 'new_coordinates', old_coordinates']")

    logging.info("bbox length is {}".format(len(bbox_complete)) )
    #bbox_complete = torch.flatten(bbox_complete, 0)
    return bbox_complete

def ocr_matrix(ocr_graph, episode_ids, sequence_length, use_ocr_as):
    ocr_tensor_complete = []
    if use_ocr_as == "input":
        if ocr_graph is not None:
            ocr_tensor = torch.zeros((27, 27))
            for ocr in ocr_graph:
                obj = ocr[0]
                contexts = ocr[1:]
                total_context_count = sum([i[1] for i in contexts])
                for context in contexts:
                    ocr_tensor[obj, context[0]] = context[1] / total_context_count
            ocr_tensor = torch.flatten(ocr_tensor)

            for idx, episode_id in enumerate(episode_ids):
                frame_paths = torch.stack([ocr_tensor for seq_len in range(sequence_length[idx])])
                ocr_tensor_complete.append(frame_paths)
                #ocr_tensor_complete.shape = (180, 27, 27)

    if use_ocr_as == "cost_func":
        if ocr_graph is not None:
            ocr_tensor = torch.zeros((27, 27))
            for ocr in ocr_graph:
                obj = ocr[0]
                contexts = ocr[1:]
                total_context_count = sum([i[1] for i in contexts])
                for context in contexts:
                    ocr_tensor[obj, context[0]] = context[1] / total_context_count
            ocr_tensor = ocr_tensor.fill_diagonal_(1) #to make the diagonal = 1 for the cost function

            for idx, episode_id in enumerate(episode_ids):
                frame_paths = torch.stack([ocr_tensor for seq_len in range(sequence_length[idx])])
                ocr_tensor_complete.append(frame_paths)

    logging.info("OCR length is {}".format(len(ocr_tensor_complete)))
    # ocr tensor.shape = [27,27]
    return ocr_tensor_complete


if __name__=='__main__':
    assert("This can not be run as a single file")