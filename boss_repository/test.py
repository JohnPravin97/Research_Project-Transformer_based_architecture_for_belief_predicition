from input_pipeline.input_dataset import Data, DataShuffle
import torch
import numpy as np
import random
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import os
import logging
from utils.utils_func import presave, CustomLoss

from configs.param import PREPROCESSED_TEST_FRAME_INPUT_PATH, LABEL_PATH, TEST_POSE_INPUT_PATH, TEST_POSE_AS_PICTURE_INPUT_PATH, TEST_GAZE_INPUT_PATH, TEST_NEW_BBOX_INPUT_PATH, TEST_OLD_BBOX_INPUT_PATH, OCR_GRAPH
from configs.param import BATCH_SIZE, EPOCHS, LR, RESHAPED_IMG_WIDTH, BBOX_AS_INPUT
from configs.param import SAVE_PATH
from configs.param import VGG_LAYERS, MODEL_NUM_LAYERS, MODEL_NUM_HEAD, MODEL_INPUT_DIM, MODEL_REPRESENTATION_DIM, MODEL_MLP_DIM
import wandb

# Parameters
test_frame_path, label_path, test_pose_path, test_pose_as_picture_path, test_gaze_path, test_new_bbox_path, test_old_bbox_path, ocr_graph_path = PREPROCESSED_TEST_FRAME_INPUT_PATH, LABEL_PATH, TEST_POSE_INPUT_PATH, TEST_POSE_AS_PICTURE_INPUT_PATH, TEST_GAZE_INPUT_PATH, TEST_NEW_BBOX_INPUT_PATH, TEST_OLD_BBOX_INPUT_PATH, OCR_GRAPH

batch_size, num_epoch, lr = BATCH_SIZE, EPOCHS, LR

class Tester():
    def __init__(self, model, device, model_type, is_ocr_cost_func, load_model_path):
        self.model = model
        self.model.load_state_dict(torch.load(os.path.join(load_model_path, 'model'), map_location=device))
        self.device = device
        self.is_ocr_cost_func = is_ocr_cost_func
        self.test_dataset = Data(test_frame_path, label_path, test_pose_path, test_gaze_path, test_new_bbox_path, test_old_bbox_path, ocr_graph_path, test_pose_as_picture_path)
        self.test_length = len(self.test_dataset)

        self.test_cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.stats = {'test': {'cls_loss': [], 'cls_acc': []}}

    def test_one_epoch(self, test_dataloader):
        test_last_loss = 0
        test_last_acc =0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_dataloader)):            
                frames, labels, left_poses, right_poses, left_gazes, right_gazes, bboxes, ocr_graphs = batch
                labels = torch.reshape(labels, (-1, 2)).to(self.device)
            
                left_belief_out, right_belief_out, ocr_out = self.model(frames, left_poses, right_poses, left_gazes, right_gazes, bboxes, ocr_graphs)
                left_belief_out = torch.reshape(left_belief_out, (-1, 27))
                right_belief_out = torch.reshape(right_belief_out, (-1, 27))

                # ACCURACY CALCULATION
                left_belief_probs = torch.softmax(left_belief_out, dim=1).argmax(dim=1)
                right_belief_probs = torch.softmax(right_belief_out, dim=1).argmax(dim=1)
                logging.info("predicted Test {}".format(left_belief_probs))
                logging.info("predicted Test {}".format(right_belief_probs))
                logging.info("Truth Test {}".format(labels[:, 0]))
                logging.info("Truth Test {}".format(labels[:, 1]))

                left_belief_target = (left_belief_probs == labels[:, 0])
                right_belief_target = (right_belief_probs == labels[:, 1])

                left_belief_acc = left_belief_target.sum().float() / float(left_belief_target.size(0))
                right_belief_acc = right_belief_target.sum().float() / float(right_belief_target.size(0))

                test_total_acc = torch.Tensor([(left_belief_acc + right_belief_acc) / 2])

                if self.is_ocr_cost_func == True:

                    # CROSS ENTROPY LOSS
                    # print(ocr_graphs[0].shape)
                    self.cus_loss = CustomLoss(ocr_graphs[0]).to(self.device)

                    test_loss = self.test_cross_entropy_loss(left_belief_out, labels[:,0]) + self.test_cross_entropy_loss(right_belief_out, labels[:,1]) + self.cus_loss(labels[:,0], labels[:,1], ocr_out)
                
                else:
                    # CROSS ENTROPY LOSS
                    test_loss = self.test_cross_entropy_loss(left_belief_out, labels[:,0]) + self.test_cross_entropy_loss(right_belief_out, labels[:,1])

                    # Gather data and report
                test_last_loss += test_loss.data.item()
                test_last_acc += test_total_acc.data.item()
                logging.info("batch {}/{} test done with cls loss={}, cls accuracy={}.".format(i+1, len(test_dataloader), test_loss.data.item(), test_total_acc.data.item()))
        
        avg_tloss =   torch.Tensor([test_last_loss  / (i + 1)])
        avg_tacc =  torch.Tensor([test_last_acc  / (i + 1)])
        logging.info('Average LOSS and ACC in Test - test_loss {}, test_acc {}'.format(avg_tloss.data.item(), avg_tacc.data.item()))

        # run WandB to monitor the parameters. Uncomment to use them
        # wandb.log({ "test_loss": avg_tloss.data.item(), "test_acc": avg_tacc.data.item()})

        self.stats['test']['cls_loss'].append(avg_tloss.data.item())
        self.stats['test']['cls_acc'].append(avg_tacc.data.item())

    def test(self):
        # get experiment ID
        experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +  '_test'
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)
        experiment_save_path = os.path.join(SAVE_PATH, experiment_id)
        os.makedirs(experiment_save_path, exist_ok=True)

        test_dataset_complete = DataShuffle(self.test_dataset, self.test_length)
        test_dataloader = DataLoader(test_dataset_complete, batch_size=batch_size, shuffle=False, num_workers=1)
        self.model.train(False)
        self.test_one_epoch(test_dataloader)

        with open(os.path.join(experiment_save_path, 'test_stat.txt'), 'w') as f:
            # f.write('{}\n'.format(cfg))
            max_test_classification_loss = self.stats['test']['cls_loss'][-1]
            max_test_classification_acc = self.stats['test']['cls_acc'][-1]
            f.write('Max test classification loss:{} & Max test classification acc:{}\n'.format(max_test_classification_loss, max_test_classification_acc))
            f.close()

if __name__ == "__main__":
    assert("This can not be run as a single file - Only called from main.py file")
