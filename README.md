# Transformer Based Architecture for Belief Prediction in Object-Context Scenarios

## Installation
Use the environment.yml file to create and install the python dependencies. 

## Dataset
In this research project, we demonstrate the performance of transformer-based models (Self-Attention Transformer Model and Hierarchical Cross Attention Transformer Model) in comprehending human beliefs through the BOSS dataset introduced by [Duan et al](https://arxiv.org/abs/2206.10665). 

## Structure
The code consists of
- This readme.
- The configs folder to set the params which are needed for the models.
- The input pipeline folder to process the input modalities in BOSS dataset and prepare them for training or testing
- The model folder to define the self-attention transformer model and hierarchical cross attention transformer architectures. 
- The train and test files to train or test the models.
- The best_model contains the details of best of the self-attention transformer model and hierarchical cross attention transformer model. 
- The utils folder to preprocess and resize frames and pose (picture), to define the custom cost function, and to define other utils func.
- The experiments folder contains results of all the models trained throughout this research project.


The main.py takes three parser arguments namely model_type (default:SelfAttentionModel), is_train_or_test (default:train), ocr_cost_func (default: false) which runs as a individual file (all the other files are linked with it). 

The input_visualization.py takes its input parameter from the param.py and used them to visualize the input. 



