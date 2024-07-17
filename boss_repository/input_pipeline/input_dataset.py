from torch.utils.data import Dataset
import os
import torch
import numpy as np
from skimage import io
import logging
from natsort import natsorted
import pickle as pkl
from torchvision import transforms
from input_pipeline.input_preprocessing import frame_func, label_func, pose_func, gaze_func, bbox_func, ocr_matrix
from configs.param import OBJECTS, RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT, BATCH_SIZE, USE_OCR_AS, POSE_AS_INPUT, BBOX_AS_INPUT

class Data(Dataset):
    def __init__(self, frame_path, label_path, pose_path=None, gaze_path=None, new_bbox_path=None, old_bbox_path=None, ocr_graph=None, pose_as_picture_path=None):
        self.frame_path = frame_path
        self.label_path = label_path
        self.pose_path = pose_path
        self.pose_as_picture_path = pose_as_picture_path
        self.pose_as_input = POSE_AS_INPUT
        self.gaze_path = gaze_path
        self.new_bbox_path = new_bbox_path
        self.old_bbox_path = old_bbox_path
        self.bbox_as_input = BBOX_AS_INPUT
        self.objects = OBJECTS
        self.ocr_graph = ocr_graph
        self.use_ocr_as = USE_OCR_AS
        self.w = RESHAPED_IMG_WIDTH
        self.h = RESHAPED_IMG_HEIGHT

        self.data = {'frame_paths': [], 'labels': [], 'left_poses': [], 'right_poses': [], 'left_gazes': [], 'right_gazes':[], 'bboxes': [], 'ocr_graph': [], "seq_length": []}

        # To get the experiment episodes from the training samples
        frame_dirs = os.listdir(self.frame_path)
        episode_ids = natsorted([frame.split(".")[0] for frame in frame_dirs])
        #episode_ids = natsorted(frame_dirs)

        # Dataloader - Frame loader 
        frame_paths_complete, sequence_length_complete = frame_func(self.frame_path, frame_dirs)
        self.data['frame_paths'], self.data['seq_length'] = frame_paths_complete, sequence_length_complete

        # Dataloader - Label loader
        labels_complete = label_func(self.label_path, episode_ids)
        self.data['labels']= labels_complete

        #Dataloader - Pose loader
        pose_left_complete, pose_right_complete = pose_func(self.pose_path, episode_ids, self.pose_as_input, self.pose_as_picture_path)
        self.data['left_poses']= pose_left_complete
        self.data['right_poses']= pose_right_complete

        #Dataloader - Gaze loader
        gaze_left_complete, gaze_right_complete = gaze_func(self.gaze_path, episode_ids)
        self.data['left_gazes']= gaze_left_complete
        self.data['right_gazes'] = gaze_right_complete

        #Dataloader - Bbox loader
        bbox_complete = bbox_func(new_bbox_path, old_bbox_path, episode_ids, self.objects, self.bbox_as_input)
        self.data['bboxes'] = bbox_complete

        #Dataloader - OCR Matrix loader
        ocr_tensor_complete = ocr_matrix(self.ocr_graph, episode_ids, sequence_length_complete, self.use_ocr_as)
        self.data['ocr_graph'] = ocr_tensor_complete

    def __len__(self):
        return len(self.data['frame_paths'])
    
    def __getitem__(self, idx):
        #Get frame
        frame_paths = self.data['frame_paths'][idx] 
        #images = torch.stack([self.transform(io.imread(i)) for i in frame_paths])
        images = torch.stack([i for i in frame_paths])

        #Get number of frames per experiment
        seq_length = self.data['seq_length'][idx] 

        #get label
        labels = (torch.tensor(self.data['labels'][idx])).permute((1,0))

        if self.pose_path is not None:
            left_pose_paths = self.data['left_poses'][idx]
            right_pose_paths = self.data['right_poses'][idx]

        if self.gaze_path is not None:
            left_gaze_paths = self.data['left_gazes'][idx]
            right_gaze_paths = self.data['right_gazes'][idx]

        if self.new_bbox_path is not None and (self.bbox_as_input == 'old_coordinates' or self.bbox_as_input == 'new_coordinates'):
            bbox_paths = self.data['bboxes'][idx]
        
        elif self.new_bbox_path is not None and (self.bbox_as_input == 'picture'):
            bboxes = self.data['bboxes'][idx]
            bbox_paths = torch.stack([i for i in bboxes])
             
        if self.ocr_graph is not None:
            ocr_graphs = self.data['ocr_graph'][idx]

        return images, labels, left_pose_paths, right_pose_paths, left_gaze_paths, right_gaze_paths, bbox_paths, ocr_graphs, seq_length

class DataShuffle(Dataset):
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.dataset_dict = {'frame_dataset': [], 'label_dataset': [], 'left_poses_dataset': [], 'right_poses_dataset': [], 'left_gaze_dataset': [], 'right_gaze_dataset': [], 'bboxes_dataset': [], 'ocr_graph_dataset': []}
        
        shuffle_list = torch.randperm(length)
        logging.info(shuffle_list)
        
        for idx, index in enumerate(shuffle_list):
            logging.info("Data Shuffle at {}/{}".format(idx+1, len(shuffle_list)))
            self.frame_dataset, self.label_dataset, self.left_pose_dataset, self.right_pose_dataset, self.left_gaze_dataset, self.right_gaze_dataset, self.bbox_dataset, self.ocr_graph_dataset, self.seq_length =  self.dataset[index]
            self.remainder = len(self.frame_dataset) % BATCH_SIZE 
            self.batch_done_at = len(self.frame_dataset) - self.remainder
            self.dataset_total_add = BATCH_SIZE - self.remainder
            #print(len(self.frame_dataset))
            #print("remainder", self.remainder)
            #print("Batch_done_at", self.batch_done_at)
            #print("dataset_total", self.dataset_total_add)

            if self.remainder == 0:
                self.dataset_dict['frame_dataset'].extend(self.frame_dataset)
                self.dataset_dict['label_dataset'].extend(self.label_dataset)
                self.dataset_dict['left_poses_dataset'].extend(self.left_pose_dataset)
                self.dataset_dict['right_poses_dataset'].extend(self.right_pose_dataset)
                self.dataset_dict['left_gaze_dataset'].extend(self.left_gaze_dataset)
                self.dataset_dict['right_gaze_dataset'].extend(self.right_gaze_dataset)
                self.dataset_dict['bboxes_dataset'].extend(self.bbox_dataset)
                self.dataset_dict['ocr_graph_dataset'].extend(self.ocr_graph_dataset) 
            
            elif self.remainder != 0:
                self.dataset_dict['frame_dataset'].extend(self.frame_dataset)
                #print("from this to batch", self.batch_done_at - self.dataset_total_add)
                self.frame_pad = self.dataset_dict['frame_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                self.dataset_dict['label_dataset'].extend(self.label_dataset)
                self.label_pad = self.dataset_dict['label_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                self.dataset_dict['left_poses_dataset'].extend(self.left_pose_dataset)
                self.left_pose_pad = self.dataset_dict['left_poses_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                self.dataset_dict['right_poses_dataset'].extend(self.right_pose_dataset)
                self.right_pose_pad = self.dataset_dict['right_poses_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                self.dataset_dict['left_gaze_dataset'].extend(self.left_gaze_dataset)
                self.left_gaze_pad = self.dataset_dict['left_gaze_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                self.dataset_dict['right_gaze_dataset'].extend(self.right_gaze_dataset)
                self.right_gaze_pad = self.dataset_dict['right_gaze_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                self.dataset_dict['bboxes_dataset'].extend(self.bbox_dataset)
                self.bbox_pad = self.dataset_dict['bboxes_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                self.dataset_dict['ocr_graph_dataset'].extend(self.ocr_graph_dataset) 
                self.ocr_pad = self.dataset_dict['ocr_graph_dataset'][self.batch_done_at - self.dataset_total_add : self.batch_done_at]

                for i in range(self.dataset_total_add):
                    self.dataset_dict['frame_dataset'].insert(self.batch_done_at+i, self.frame_pad[i])
                    self.dataset_dict['label_dataset'].insert(self.batch_done_at+i, self.label_pad[i])
                    self.dataset_dict['left_poses_dataset'].insert(self.batch_done_at+i, self.left_pose_pad[i])
                    self.dataset_dict['right_poses_dataset'].insert(self.batch_done_at+i, self.right_pose_pad[i])
                    self.dataset_dict['left_gaze_dataset'].insert(self.batch_done_at+i, self.left_gaze_pad[i])
                    self.dataset_dict['right_gaze_dataset'].insert(self.batch_done_at+i, self.right_gaze_pad[i])
                    self.dataset_dict['bboxes_dataset'].insert(self.batch_done_at+i, self.bbox_pad[i])
                    self.dataset_dict['ocr_graph_dataset'].insert(self.batch_done_at+i, self.ocr_pad[i])

            #print(len(self.dataset_dict['bboxes_dataset']))
                
            '''
            elif self.remainder != 0:
                self.frame_zero_pad = torch.zeros_like(self.frame_dataset[0])
                self.dataset_dict['frame_dataset'].extend(self.frame_dataset)
                
                self.label_zero_pad = torch.zeros_like(self.label_dataset[0])
                self.dataset_dict['label_dataset'].extend(self.label_dataset)

                self.pose_zero_pad = torch.zeros_like(self.left_pose_dataset[0])
                self.dataset_dict['left_poses_dataset'].extend(self.left_pose_dataset)
                self.dataset_dict['right_poses_dataset'].extend(self.right_pose_dataset)

                self.gaze_zero_pad = torch.zeros_like(self.left_gaze_dataset[0])
                self.dataset_dict['left_gaze_dataset'].extend(self.left_gaze_dataset)
                self.dataset_dict['right_gaze_dataset'].extend(self.right_gaze_dataset)

                self.bbox_zero_pad = torch.zeros_like(self.bbox_dataset[0])
                self.dataset_dict['bboxes_dataset'].extend(self.bbox_dataset)

                self.ocr_zero_pad = torch.zeros_like(self.ocr_graph_dataset[0])
                self.dataset_dict['ocr_graph_dataset'].extend(self.ocr_graph_dataset) 

                for i in range(self.dataset_total_add):
                    self.dataset_dict['frame_dataset'].append(self.frame_zero_pad)
                    self.dataset_dict['label_dataset'].append(self.label_zero_pad)
                    self.dataset_dict['left_poses_dataset'].append(self.pose_zero_pad)
                    self.dataset_dict['right_poses_dataset'].append(self.pose_zero_pad)
                    self.dataset_dict['left_gaze_dataset'].append(self.gaze_zero_pad)
                    self.dataset_dict['right_gaze_dataset'].append(self.gaze_zero_pad)
                    self.dataset_dict['bboxes_dataset'].append(self.bbox_zero_pad)
                    self.dataset_dict['ocr_graph_dataset'].append(self.ocr_zero_pad) 
            ''' 
        logging.info("Total Images {}".format(len(self.dataset_dict['bboxes_dataset'])))

    def __len__(self):
        return len(self.dataset_dict['frame_dataset'])
    
    def __getitem__(self, idx):
        # Complete frame from all the experiments
        frame_dataset_complete = self.dataset_dict['frame_dataset'][idx] 
        
        # Complete labels from all the experiments
        label_dataset_complete = self.dataset_dict['label_dataset'][idx] 

        # Complete poses from all the experiments
        left_poses_dataset_complete = self.dataset_dict['left_poses_dataset'][idx] 
        right_poses_dataset_complete = self.dataset_dict['right_poses_dataset'][idx] 

        # Complete gazes from all the experiments
        left_gaze_dataset_complete = self.dataset_dict['left_gaze_dataset'][idx] 
        right_gaze_dataset_complete = self.dataset_dict['right_gaze_dataset'][idx] 

        # Complete bbox from all the experiments
        bboxes_dataset_complete = self.dataset_dict['bboxes_dataset'][idx] 

        # Complete OCR from all the experiments
        ocr_graph_dataset_complete = self.dataset_dict['ocr_graph_dataset'][idx] 
       
        return frame_dataset_complete, label_dataset_complete, left_poses_dataset_complete, right_poses_dataset_complete, left_gaze_dataset_complete, right_gaze_dataset_complete, bboxes_dataset_complete, ocr_graph_dataset_complete

if __name__=='__main__':
    assert("This can not be run as a single file")