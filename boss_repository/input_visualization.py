#import argparse
import os
import pickle
import cv2
import imageio
import numpy as np
from treelib import Tree
from configs.param import EXP_NO, INPUT_MODALITIES, TRAIN_FRAME_INPUT_PATH, TRAIN_OLD_BBOX_INPUT_PATH, TRAIN_NEW_BBOX_INPUT_PATH, BBOX_INPUT, TRAIN_POSE_INPUT_PATH, LIMBS_VIZ, N_JOINTS

from configs.param import OCR_GRAPH, OLD_OCR_GRAPH, OBJECTS, OUTPUT_GIF_PATH

def ocr_matrix_visualize(ocr_graph, output_path, objects):
    """
    This function helps to visualize the OCR tree collected from the training samples

    Input Parameters:
        ocr_graph -> list containing the ocr graph 
        output_path -> path to store the OCR tree

    Return:
        None
    
    """    
    tree = Tree()
    tree.create_node("OCR_Matrix", "ocr_matrix")
    subnode_id, sub_subnode_id=0, 100
    for idx in ocr_graph:
        root = idx[0]
        tree.create_node(objects[root], objects[root].capitalize(), parent="ocr_matrix")
        for node in idx:
            if node !=root:
                subnode_id+=1
                sub_subnode_id+=1
                tree.create_node(objects[node[0]], str(subnode_id), parent=objects[root].capitalize())
                tree.create_node(node[1], str(sub_subnode_id), parent=str(subnode_id)) 
    tree.save2file(os.path.join(output_path, 'ocr_matrix.txt'))
    print("done creating the OCR tree and store at " + output_path)

def yolo_to_x_y(x_center, y_center, x_width, y_height, width, height):
    """
    This function helps the new_bbox_visualize function

    Input Parameters:
        x_center, y_center, x_width, y_height -> YOLO bbox annotation format. Refer below link for more details
        #https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/\

        width, height - size of the image on a loop

    Return:
       x1, y1, x2, y2
    
    """
    x_center *= width
    y_center *= height
    x_width *= width
    y_height *= height
    x_width /= 2.0
    y_height /= 2.0
    return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)

def new_bbox_visualize(frame_input_path, bbox_input_path, objects, output_path):
    """
    This function helps to visualize the new bbox data collected from the YOLOv5 model

    Input Parameters:
        frame_input_path -> list containing all the frames of the given experiment
        bbox_input_path -> list containing all the new bbox of the given experiment
        output_path -> path to store the bbox gif file for the given experiment

    Return:
        None
    
    """
    frames=[]

    #to collect all the frames of the experiment
    frame_new_path = os.listdir(frame_input_path)
    frame_new_path = sorted(frame_new_path, key=len)

    #to collect all the bbox of the experiment
    bbox_new_path = os.listdir(bbox_input_path)
    bbox_new_path = sorted(bbox_new_path, key=len)

    for i in range(len(frame_new_path)):
        image = cv2.imread(os.path.join(frame_input_path, frame_new_path[i]))
        #image = cv2.resize(image, (640, 480) , interpolation = cv2.INTER_AREA)
        height, width, _ = image.shape
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with open(os.path.join(bbox_input_path, bbox_new_path[i]), 'r') as fp:
            content = fp.readlines()
        for line in content:
            values_str = line.split()
            class_index, x_center, y_center, x_width, y_height = map(float, values_str)
            x1, y1, x2, y2 = yolo_to_x_y(x_center, y_center, x_width, y_height, width, height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(image, objects[int(class_index)], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(image, "Object Detection", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    
    #to create the gif file
    imageio.mimsave(os.path.join(output_path, 'new_bbox_of_exp_'+ EXP_NO +'.gif'), frames, "GIF")
    print("done creating the gif and store at " + output_path)

def old_bbox_visualize(frame_input_path, bbox_input_path, output_path):
    """
    This function helps to visualize the old bbox data collected from the detecto model by author duan et al

    Input Parameters:
        frame_input_path -> list containing all the frames of the given experiment
        bbox_input_path -> list containing all the new bbox of the given experiment
        output_path -> path to store the bbox gif file for the given experiment

    Return:
        None
    
    """
    frames=[]
    #to collect all the frames of the experiment
    frame_new_path_list = os.listdir(frame_input_path)
    frame_new_path_list = sorted(frame_new_path_list, key=len)

    #to collect all the corresponding bbox of the experiment
    with open(bbox_input_path, 'rb') as Bbox_fp:
        bboxes_i = pickle.load(Bbox_fp)

    #to mark the bbox onto the frame
    for i in range(len(frame_new_path_list)):
        image = cv2.imread(os.path.join(frame_input_path, frame_new_path_list[i]))
        for j in range(len(bboxes_i[i][0])):
            cv2.rectangle(image, (int(bboxes_i[i][1][j][0]), int(bboxes_i[i][1][j][1])), (int(bboxes_i[i][1][j][2]),int(bboxes_i[i][1][j][3])), (255,0,0), 2)
            cv2.putText(image, bboxes_i[i][0][j], (int(bboxes_i[i][1][j][0]), int(bboxes_i[i][1][j][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(image, "Object Detection", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    
    #to create the gif file
    imageio.mimsave(os.path.join(output_path, 'bbox_of_exp_'+ EXP_NO +'.gif'), frames, "GIF")
    print("done creating the gif and store at " + output_path)
    
def pose_visualize(frame_input_path, pose_input_path, output_path, limbs, n_joints, remove_less_conf=False):
    """
    This function helps to visualize the pose data collected from the OpenPose model by author duan et al

    Input Parameters:
        frame_input_path -> list containing all the frames of the given experiment
        pose_input_path -> list containing all the new bbox of the given experiment
        output_path -> path to store the bbox gif file for the given experiment
        limbs -> list containing the connection betweent the joints e.g head to neck
        #https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
        n_joints -> number of joints in total
        remove_less_conf -> BOOLEAN input to remove the joints connection if confident value is less than 0.5

    Return:
        None
    
    """
    
    frames=[]
    #to collect the experiment number
    exp = frame_input_path.split("\\")[-1]
    
    #to collect all the frames of the experiment
    frame_new_path_list = os.listdir(frame_input_path)
    frame_new_path_list = sorted(frame_new_path_list, key=len)

    #to collect all the pose of the experiment
    poses_new_path_list = os.listdir(pose_input_path)
    poses_new_path_list = sorted(poses_new_path_list, key=len)

    #to mark the bbox onto the frame
    for i in range(len(frame_new_path_list)):
        #Frames
        image = cv2.imread(os.path.join(frame_input_path, frame_new_path_list[i]))
        #height, width, _ = image.shape
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #Poses
        fs = cv2.FileStorage(os.path.join(pose_input_path, poses_new_path_list[i]), cv2.FILE_STORAGE_READ)
        poses=[]
        if np.array(fs.getNode("pose_0").mat()).shape != (2,25,3):
            poses.append(fs.getNode("pose_0").mat()[:2,:,:])
        else:
            poses.append(fs.getNode("pose_0").mat())
            
        # Radius of circle
        radius = 5

        # Blue color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2
        
        
        if remove_less_conf:
            # To remove the joints with less conf to zero
            for i in range(len(poses[0][0])):
                if poses[0][0][i][2] <= 0.5:
                    poses[0][0][i][0] = 0
                    poses[0][0][i][1] = 0
                    poses[0][0][i][2] = 0
        
        
        # to mark the circle which denotes the joints
        for i in range(n_joints):
            center_coordinates = (int(poses[0][1][i][0]), int(poses[0][1][i][1]))
            image = cv2.circle(image, center_coordinates, radius, color, thickness)
            

            center_coordinates = (int(poses[0][0][i][0]), int(poses[0][0][i][1]))
            image = cv2.circle(image, center_coordinates, radius, color, thickness)
        
        #to mark the joints between the nodes with a line
        for start_point, end_point in limbs:
            start_left = (int(poses[0][1][start_point][0]), int(poses[0][1][start_point][1]))
            end_left = (int(poses[0][1][end_point][0]), int(poses[0][1][end_point][1]))
            if start_left == (0,0) or end_left ==(0,0):
                continue
            else:
                cv2.line(image, start_left, end_left, color, thickness) 
                
        for start_point, end_point in limbs:
            start_right = (int(poses[0][0][start_point][0]), int(poses[0][0][start_point][1]))
            end_right = (int(poses[0][0][end_point][0]), int(poses[0][0][end_point][1]))
            
            if start_right == (0,0) or end_right ==(0,0):
                continue
            else:
                cv2.line(image, start_right, end_right, color, thickness) 
            
        cv2.putText(image, "Pose Estimation", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)

    #to create the gif file
    imageio.mimsave(os.path.join(output_path, 'poses_of_exp_'+ EXP_NO + '.gif'), frames, "GIF")
    print("done creating the gif and store at " + output_path)


def main_visualize():

    input = INPUT_MODALITIES

    frame_input_path = os.path.join(TRAIN_FRAME_INPUT_PATH, EXP_NO)

    old_bbox_input_path = os.path.join(TRAIN_OLD_BBOX_INPUT_PATH, EXP_NO)
    new_bbox_input_path = os.path.join(TRAIN_NEW_BBOX_INPUT_PATH, "labels", EXP_NO)
    bbox_input = BBOX_INPUT

    pose_input_path = TRAIN_POSE_INPUT_PATH +"//" + EXP_NO
    limbs = LIMBS_VIZ
    n_joints = N_JOINTS

    ocr_graph = OCR_GRAPH
    old_ocr_graph = OLD_OCR_GRAPH
    objects = OBJECTS

    output_path = OUTPUT_GIF_PATH

    # Conditions for the different inputs visualization
    if input == "bbox" and bbox_input == "old":
        old_bbox_visualize(frame_input_path, old_bbox_input_path, output_path)
    elif input == "bbox" and bbox_input == "new":
        new_bbox_visualize(frame_input_path, new_bbox_input_path, objects, output_path)
    elif input == "pose":
        pose_visualize(frame_input_path, pose_input_path, output_path, limbs, n_joints)
    elif input =="ocr":
        ocr_matrix_visualize(ocr_graph, output_path, objects)
    else:
        print("Invalid inputs")

    return None


if __name__=='__main__':
    main_visualize()
