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
from configs.param import PREPROCESSED_TRAIN_FRAME_INPUT_PATH, LABEL_PATH, TRAIN_POSE_INPUT_PATH, TRAIN_POSE_AS_PICTURE_INPUT_PATH, TRAIN_GAZE_INPUT_PATH, TRAIN_NEW_BBOX_INPUT_PATH, TRAIN_OLD_BBOX_INPUT_PATH, OCR_GRAPH
from configs.param import PREPROCESSED_VAL_FRAME_INPUT_PATH, LABEL_PATH, VAL_POSE_INPUT_PATH, VAL_POSE_AS_PICTURE_INPUT_PATH, VAL_GAZE_INPUT_PATH, VAL_NEW_BBOX_INPUT_PATH, VAL_OLD_BBOX_INPUT_PATH, OCR_GRAPH
from configs.param import BATCH_SIZE, EPOCHS, LR, RESHAPED_IMG_WIDTH, BBOX_AS_INPUT, POSE_AS_INPUT, LAMBD, VGG_LAYERS_BBOX, WEIGHT_DECAY
from configs.param import SAVE_PATH, MODEL_ORIGINAL_INPUT_LENGTH_METHOD_1, MODEL_ORIGINAL_INPUT_LENGTH_METHOD_2
from configs.param import VGG_LAYERS, MODEL_NUM_LAYERS, MODEL_NUM_HEAD, MODEL_INPUT_DIM, MODEL_REPRESENTATION_DIM, MODEL_MLP_DIM
import wandb
import neptune


## Parameters Definition
train_frame_path, label_path, train_pose_path, train_pose_as_picture_path, train_gaze_path, train_new_bbox_path, train_old_bbox_path, ocr_graph_path = PREPROCESSED_TRAIN_FRAME_INPUT_PATH, LABEL_PATH, TRAIN_POSE_INPUT_PATH, TRAIN_POSE_AS_PICTURE_INPUT_PATH, TRAIN_GAZE_INPUT_PATH, TRAIN_NEW_BBOX_INPUT_PATH, TRAIN_OLD_BBOX_INPUT_PATH, OCR_GRAPH

val_frame_path, label_path, val_pose_path, val_pose_as_picture_path, val_gaze_path, val_new_bbox_path, val_old_bbox_path, ocr_graph_path = PREPROCESSED_VAL_FRAME_INPUT_PATH, LABEL_PATH, VAL_POSE_INPUT_PATH, VAL_POSE_AS_PICTURE_INPUT_PATH, VAL_GAZE_INPUT_PATH, VAL_NEW_BBOX_INPUT_PATH, VAL_OLD_BBOX_INPUT_PATH, OCR_GRAPH

batch_size, num_epoch, lr = BATCH_SIZE, EPOCHS, LR

class Trainer():
    def __init__(self, model, device, model_type, is_ocr_cost_func):
        #presave()
        self.model = model
        self.device = device
        self.is_ocr_cost_func = is_ocr_cost_func

        self.train_dataset = Data(train_frame_path, label_path, train_pose_path, train_gaze_path, train_new_bbox_path, train_old_bbox_path, ocr_graph_path, train_pose_as_picture_path)
        self.train_length = len(self.train_dataset)
        self.val_dataset = Data(val_frame_path, label_path, val_pose_path, val_gaze_path, val_new_bbox_path, val_old_bbox_path, ocr_graph_path, val_pose_as_picture_path)
        self.val_length = len(self.val_dataset)

        # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay = WEIGHT_DECAY)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        #self.train_cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.2).to(self.device)
        self.train_cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.val_cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)

        self.stats = {'train': {'cls_loss': [], 'cls_acc': []}, 'val': {'cls_loss': [], 'cls_acc': []}}
        self.lambda_value = LAMBD

        # run Neptune.AI to monitor the parameters. Uncomment to use them
        # self.run_neptune = neptune.init_run(project="your pooject name token here",  api_token="your API token here")  # your API token here
        # neptune_params = {"model_type": model_type, "batch_size": batch_size, "num_epoch":num_epoch, "Image Resized as":RESHAPED_IMG_WIDTH, "MODEL_ORIGINAL_INPUT_LENGTH_METHOD_1" : MODEL_ORIGINAL_INPUT_LENGTH_METHOD_1, "MODEL_ORIGINAL_INPUT_LENGTH_METHOD_2": MODEL_ORIGINAL_INPUT_LENGTH_METHOD_2, "is_ocr_cost_func": is_ocr_cost_func, "Learning Rate":lr, "VGG Layers": VGG_LAYERS, "Model Num Layers":MODEL_NUM_LAYERS , "Lambda": self.lambda_value, "Model Num Head": MODEL_NUM_HEAD, "Model Input Dim": MODEL_INPUT_DIM, "Model Representation Dim": MODEL_REPRESENTATION_DIM, "Model MLP dim":MODEL_MLP_DIM, "BBOX Input": BBOX_AS_INPUT, "Pose Input": POSE_AS_INPUT, "Bbox_Vgg_Layers": VGG_LAYERS_BBOX, "Learning Rate": lr, "Weight_Decay": WEIGHT_DECAY}
        # self.run_neptune["parameters"] = neptune_params

        # run WandB to monitor the parameters. Uncomment to use them
        #wandb.init(project=model_type, config={"model_type": model_type, "batch_size": batch_size, "num_epoch":num_epoch, "Image Resized as":RESHAPED_IMG_WIDTH, "is_ocr_cost_func": is_ocr_cost_func, "Learning Rate":lr, "VGG Layers": VGG_LAYERS, "Model Num Layers":MODEL_NUM_LAYERS , "Lambda": self.lambda_value, "Model Num Head": MODEL_NUM_HEAD, "Model Input Dim": MODEL_INPUT_DIM, "Model Representation Dim": MODEL_REPRESENTATION_DIM, "Model MLP dim":MODEL_MLP_DIM, "BBOX Input": BBOX_AS_INPUT, "Pose Input": POSE_AS_INPUT, "Bbox_Vgg_Layers": VGG_LAYERS_BBOX, "Learning Rate": lr})
        #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=pad_collate)
                      
    def train_one_epoch(self, epoch, train_dataloader):
        last_loss = 0
        last_acc=0
        #last_accuracy = []
        for i, batch in tqdm(enumerate(train_dataloader)):
            self.optimizer.zero_grad()
            frames, labels, left_poses, right_poses, left_gazes, right_gazes, bboxes, ocr_graphs = batch

            labels = torch.reshape(labels, (-1, 2)).to(self.device)

            left_belief_out, right_belief_out, ocr_out = self.model(frames, left_poses, right_poses, left_gazes, right_gazes, bboxes, ocr_graphs)
            left_belief_out = torch.reshape(left_belief_out, (-1, 27))
            right_belief_out = torch.reshape(right_belief_out, (-1, 27))

            # ACCURACY CALCULATION
            left_belief_probs = torch.softmax(left_belief_out, dim=1).argmax(dim=1)
            right_belief_probs = torch.softmax(right_belief_out, dim=1).argmax(dim=1)
            # logging.info("predicted{}".format(left_belief_probs))
            # logging.info("predicted {}".format(right_belief_probs))
            # logging.info("Truth {}".format(labels[:, 0]))
            # logging.info("Truth {}".format(labels[:, 1]))

            left_belief_target = (left_belief_probs == labels[:, 0])
            right_belief_target = (right_belief_probs == labels[:, 1])

            left_belief_acc = left_belief_target.sum().float() / float(left_belief_target.size(0))
            right_belief_acc = right_belief_target.sum().float() / float(right_belief_target.size(0))

            total_acc = torch.Tensor([(left_belief_acc + right_belief_acc) / 2])

            if self.is_ocr_cost_func == True:
                # CROSS ENTROPY LOSS
                self.cus_loss = CustomLoss(ocr_graphs[0]).to(self.device)

                loss = self.train_cross_entropy_loss(left_belief_out, labels[:,0]) + self.train_cross_entropy_loss(right_belief_out, labels[:,1]) + (self.lambda_value * (self.cus_loss(labels[:,0], labels[:,1], ocr_out)))
            
            else:
                # CROSS ENTROPY LOSS
                loss = self.train_cross_entropy_loss(left_belief_out, labels[:,0]) + self.train_cross_entropy_loss(right_belief_out, labels[:,1])

            loss.backward()  
            #loss.backward(retain_graph=True)
            self.optimizer.step() 
            # Gather data and report
            last_loss += loss.data.item()
            last_acc += total_acc.data.item()
            logging.info("Epoch {}/{} batch {}/{} training done with cls loss={}, cls accuracy={}.".format(epoch+1, num_epoch, i+1, len(train_dataloader), loss.data.item(), total_acc.data.item()))
        
        avg_loss =  torch.Tensor([last_loss  / (i + 1)])
        avg_acc = torch.Tensor([last_acc / (i+1)])
        logging.info('Average LOSS and Acc Per epoch - train_loss {}, train_acc {}'.format(avg_loss.data.item(), avg_acc.data.item()))
        
        # run Neptune.AI to monitor the parameters. Uncomment to use them
        # self.run_neptune["train/epoch"].append(epoch+1)
        # self.run_neptune["train/train_loss"].append(avg_loss.data.item())
        # self.run_neptune["train/train_acc"].append(avg_acc.data.item())

        # run WandB to monitor the parameters. Uncomment to use them
        # wandb.log({"Epoch": epoch+1, "train_loss": avg_loss.data.item(), "train_acc": avg_acc.data.item()})

        self.stats['train']['cls_loss'].append(avg_loss.data.item())
        self.stats['train']['cls_acc'].append(avg_acc.data.item())

    
    def val_one_epoch(self, epoch, val_dataloader):
        val_last_loss = 0
        val_last_acc = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_dataloader)):            
                frames, labels, left_poses, right_poses, left_gazes, right_gazes, bboxes, ocr_graphs = batch
                labels = torch.reshape(labels, (-1, 2)).to(self.device)

                #logging.info("OUTPUT OF  THE MODEL {}".format(labels))
                left_belief_out, right_belief_out, ocr_out = self.model(frames, left_poses, right_poses, left_gazes, right_gazes, bboxes, ocr_graphs)
                left_belief_out = torch.reshape(left_belief_out, (-1, 27))
                right_belief_out = torch.reshape(right_belief_out, (-1, 27))
                #logging.info("OUTPUT OF  THE MODEL {}".format(left_belief_out))
                #logging.info("OUTPUT OF  THE MODEL {}".format(right_belief_out))

                # ACCURACY CALCULATION
                left_belief_probs = torch.softmax(left_belief_out, dim=1).argmax(dim=1)
                right_belief_probs = torch.softmax(right_belief_out, dim=1).argmax(dim=1)
                logging.info("predicted {}".format(left_belief_probs))
                logging.info("predicted {}".format(right_belief_probs))
                logging.info("Truth {}".format(labels[:, 0]))
                logging.info("Truth {}".format(labels[:, 1]))

                left_belief_target = (left_belief_probs == labels[:, 0])
                right_belief_target = (right_belief_probs == labels[:, 1])

                left_belief_acc = left_belief_target.sum().float() / float(left_belief_target.size(0))
                right_belief_acc = right_belief_target.sum().float() / float(right_belief_target.size(0))

                val_total_acc = torch.Tensor([(left_belief_acc + right_belief_acc) / 2])

                if self.is_ocr_cost_func == True:

                    # CROSS ENTROPY LOSS
                    #print(ocr_graphs[0].shape)
                    logging.info("Running With OCR Loss")
                    self.cus_loss = CustomLoss(ocr_graphs[0]).to(self.device)

                    val_loss = self.val_cross_entropy_loss(left_belief_out, labels[:,0]) + self.val_cross_entropy_loss(right_belief_out, labels[:,1]) + (self.lambda_value * (self.cus_loss(labels[:,0], labels[:,1], ocr_out)))
                
                else:
                    # CROSS ENTROPY LOSS
                    val_loss = self.val_cross_entropy_loss(left_belief_out, labels[:,0]) + self.val_cross_entropy_loss(right_belief_out, labels[:,1])

                    # Gather data and report
                val_last_loss += val_loss.data.item()
                val_last_acc += val_total_acc.data.item()
                logging.info("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}.".format(epoch+1, num_epoch, i+1, len(val_dataloader), val_loss.data.item(), val_total_acc.data.item()))
        
        avg_vloss =   torch.Tensor([val_last_loss  / (i + 1)])
        avg_vacc =  torch.Tensor([val_last_acc  / (i + 1)])

        logging.info('Average LOSS and ACC Per epoch - val_loss {}, val_acc {}'.format(avg_vloss.data.item(), avg_vacc.data.item()))

        # run Neptune.AI to monitor the parameters. Uncomment to use them
        # if torch.isnan(avg_vloss):      
        #     self.run_neptune["val/epoch"].append(epoch+1)
        #     self.run_neptune["val/val_loss"].append(10)
        #     self.run_neptune["val/val_acc"].append(avg_vacc.data.item())
        # else:     
        #     self.run_neptune["val/epoch"].append(epoch+1)
        #     self.run_neptune["val/val_loss"].append(avg_vloss.data.item())
        #     self.run_neptune["val/val_acc"].append(avg_vacc.data.item())

        # run WandB to monitor the parameters. Uncomment to use them
        #wandb.log({"Epoch": epoch+1, "val_loss": avg_vloss.data.item(), "val_acc": avg_vacc.data.item()})

        self.stats['val']['cls_loss'].append(avg_vloss.data.item())
        self.stats['val']['cls_acc'].append(avg_vacc.data.item())

    def train(self):
        max_val_classification_acc = 0
        max_val_classification_epoch = None

        # get experiment ID
        experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +  '_train'
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)
        experiment_save_path = os.path.join(SAVE_PATH, experiment_id)
        os.makedirs(experiment_save_path, exist_ok=True)
        
        for epoch in range(num_epoch):
            train_dataset_complete = DataShuffle(self.train_dataset, self.train_length)
            train_dataloader = DataLoader(train_dataset_complete, batch_size=batch_size, shuffle=False, num_workers=1)
            val_dataset_complete = DataShuffle(self.val_dataset, self.val_length)
            val_dataloader = DataLoader(val_dataset_complete, batch_size=batch_size, shuffle=False, num_workers=1)

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            self.train_one_epoch(epoch, train_dataloader)
            # We don't need gradients on to do reporting
            self.model.train(False)
            self.val_one_epoch(epoch, val_dataloader)
            #print("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}.".format(epoch+1, num_epoch, j+1, len(val_dataloader), loss.data.item(), batch_val_acc))

            # check for best stat/model using validation accuracy
            if self.stats['val']['cls_acc'][-1] >= max_val_classification_acc:
                max_val_classification_acc = self.stats['val']['cls_acc'][-1]
                max_val_classification_epoch = epoch+1
                max_train_classification_acc = self.stats['train']['cls_acc'][-1]
                max_train_classification_epoch = epoch+1
                #os.makedirs(os.path.join(experiment_save_path, 'model'), exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(experiment_save_path, 'model'))

            with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
                # f.write('{}\n'.format(cfg))
                f.write('{}\n'.format(self.stats))
                f.write('Max val classification acc: epoch {}, {}\n'.format(max_val_classification_epoch, max_val_classification_acc))
                f.write('Max train classification acc: epoch {}, {}\n'.format(max_train_classification_epoch, max_train_classification_acc))
                f.close()

        # run WandB to monitor the parameters. Uncomment to use them
        # self.run_neptune.stop()
        
        return experiment_save_path


if __name__ == "__main__":
    assert("This can not be run as a single file - Only called from main.py file")