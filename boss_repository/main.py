from absl import app, flags
import torch
from utils import utils_params
from configs.param import MODEL_NUM_LAYERS, MODEL_INPUT_DIM, MODEL_NUM_HEAD, MODEL_MLP_DIM, MODEL_REPRESENTATION_DIM, MODEL_NUM_CLASSES, MODEL_ORIGINAL_INPUT_LENGTH_METHOD_1, MODEL_ORIGINAL_INPUT_LENGTH_METHOD_2
from configs.param import MODEL_ORIGINAL_INPUT_CA_LENGTH_METHOD_1
from configs.param import SA_LOAD_MODEL_PATH, CA_LOAD_MODEL_PATH
from model.architectures import SelfAttentionViT as SaViT
from model.architectures import CrossAttentionViT as CaViT
from train import Trainer
from test import Tester
import os
import logging
import neptune
from utils.utils_func import summary_table_plotter, presave_pose_as_picture
import argparse

# MODEL PARAMETERS
num_layers, input_dim, num_heads, mlp_dim, representation_dim, num_classes, original_input_length_M1, original_input_length_M2  = MODEL_NUM_LAYERS, MODEL_INPUT_DIM, MODEL_NUM_HEAD, MODEL_MLP_DIM, MODEL_REPRESENTATION_DIM, MODEL_NUM_CLASSES, MODEL_ORIGINAL_INPUT_LENGTH_METHOD_1, MODEL_ORIGINAL_INPUT_LENGTH_METHOD_2
original_ca_input_length_M1 = MODEL_ORIGINAL_INPUT_CA_LENGTH_METHOD_1

# DEFINING THE DEVICE
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else 'cpu')

def main():
    
    #Argument Parser
    ap = argparse.ArgumentParser(description= "Transformer based Arch for BOSS Dataset", epilog="Thanks for using me")
    #arguments
    ap.add_argument(
        "model_type",
        help = "Model Name choose from [SelfAttentionModel, CrossAttentionModel]", 
        default = "SelfAttentionModel",
        type = str
    )

    ap.add_argument("is_train_or_test",
        help = "choose from [train, test]", 
        default = "train",
        type = str
    )


    ap.add_argument("-v",
        "--ocr_cost_func", 
        help = "OCR matric as cost function or not",
        action='store_true'
    )


    args = ap.parse_args()
    model_type = args.model_type
    ocr_cost_func = args.ocr_cost_func
    is_train_or_test = args.is_train_or_test

    # To preprocess and resize the frame - Run once
    # frame_save_folder = # the location to save the generated pose as pictures. 
    # _ = presave(frame_save_folder)

    # To generate pose as picture - Run once 
    # pose_save_folder = # the location to save the generated pose as pictures. 
    # _ = presave_pose_as_picture(pose_save_folder)

    #initiating wandb
    #wandb.login(key='GIVE YOUR KEY')

    ## TO TRAIN THE MODEL 

    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_params.set_loggers(run_paths['path_logs_train'], logging.INFO)

    logging.info("Model {}".format(model_type))
    logging.info("OCR_Cus_Function {}".format(ocr_cost_func))
    
    if model_type == 'SelfAttentionModel':
        model = SaViT(original_input_length_M1,
                        num_layers,
                        num_heads,
                        input_dim,
                        mlp_dim,
                        representation_dim,
                        num_classes, 
                        device, 
                        is_ocr_cost_func = ocr_cost_func).to(device)
        #wandb.init(project="SA-1", config={"model_type": model_type, "batch_size": batch_size, "num_epoch":num_epoch, "is_ocr_cost_func": is_ocr_cost_func, "Learning Rate":lr})
        logging.info("Training Started")

        # Train and test the model
        if is_train_or_test == "train":
            #Train
            train = Trainer(model = model, device= device, model_type = model_type, is_ocr_cost_func = ocr_cost_func)
            load_model_path = train.train()

            # To get the Summary of the Model Parameters
            summary_table_plotter(model, load_model_path)

            #Test 
            logging.info("Test Started")
            test = Tester(model = model, device= device, model_type = model_type, is_ocr_cost_func = ocr_cost_func, load_model_path = load_model_path)
            test.test()

        # Only test the presaved model
        elif is_train_or_test == "test":

            load_model_path = SA_LOAD_MODEL_PATH

            #Test 
            logging.info("Test Started")
            test = Tester(model = model, device= device, model_type = model_type, is_ocr_cost_func = ocr_cost_func, load_model_path = load_model_path)
            test.test()


        else:
            raise Exception("Invalid Input")
    
    elif model_type == 'CrossAttentionModel':
        model = CaViT(original_ca_input_length_M1,
                        original_ca_input_length_M1,
                        num_layers,
                        num_heads,
                        input_dim,
                        mlp_dim,
                        representation_dim,
                        num_classes, 
                        device, 
                        is_ocr_cost_func = ocr_cost_func).to(device)
        
        #wandb.init(project="SA-1", config={"model_type": model_type, "batch_size": batch_size, "num_epoch":num_epoch, "is_ocr_cost_func": is_ocr_cost_func, "Learning Rate":lr})
        

        # Train and test the model
        if is_train_or_test == "train":
            logging.info("Training Started")

            train = Trainer(model = model, device= device, model_type = model_type, is_ocr_cost_func = ocr_cost_func)
            load_model_path = train.train()

            # To get the Summary of the Model Parameters
            summary_table_plotter(model, load_model_path)

            #Test 
            logging.info("Test Started")
            test = Tester(model = model, device= device, model_type = model_type, is_ocr_cost_func = ocr_cost_func, load_model_path = load_model_path)
            test.test()
        
        # Only test the presaved model
        elif is_train_or_test == "test":

            load_model_path = CA_LOAD_MODEL_PATH

            #Test 
            logging.info("Test Started")
            test = Tester(model = model, device= device, model_type = model_type, is_ocr_cost_func = ocr_cost_func, load_model_path = load_model_path)
            test.test()


        else:
            raise Exception("Invalid Input")
    
    else:
        raise Exception("Invalid Input")


if __name__ == "__main__":
    #app.run(main)
    main() 