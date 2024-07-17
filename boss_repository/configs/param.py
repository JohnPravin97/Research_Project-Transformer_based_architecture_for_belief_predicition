"""
Various important parameters of our model and training procedure.
NOTE: THE LOCATION OF FEW INPUTS ARE LEFT AS THE ORIGINAL LOCATION IN hci-gpuws4. PLEASE CHANGE IT TO YOUR PATH
"""
import torch 

## IMAGE PARAMETERS
ORIGINAL_IMG_WIDTH, ORIGINAL_IMG_HEIGHT  = 1920, 1088
RESHAPED_IMG_WIDTH, RESHAPED_IMG_HEIGHT = 128, 128


LABEL_PATH = r"/projects/arockiasamy/datasets/boss/outfile"
OUTPUT_GIF_PATH = r"/projects/arockiasamy/infotech2023_john/visualization_gifs"

## TRAIN PATHS
TRAIN_FRAME_INPUT_PATH =r"/datasets/external/boss/train/frame" 
PREPROCESSED_TRAIN_FRAME_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_128/train/frame" 
#PREPROCESSED_TRAIN_FRAME_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_256/train/frame"
TRAIN_OLD_BBOX_INPUT_PATH = r"/datasets/external/boss/train/bbox"
TRAIN_NEW_BBOX_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_8/train/new_bbox"
#TRAIN_NEW_BBOX_INPUT_PATH = r"/datasets/external/boss/train/new_bbox"
TRAIN_POSE_INPUT_PATH = r"/datasets/external/boss/train/pose"
TRAIN_POSE_AS_PICTURE_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_pose_16/train"
TRAIN_GAZE_INPUT_PATH = r"/datasets/external/boss/train/gaze"

## TEST PATHS
TEST_FRAME_INPUT_PATH =r"/datasets/external/boss/test/frame" 
PREPROCESSED_TEST_FRAME_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_128/test/frame"
#PREPROCESSED_TEST_FRAME_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_256/test/frame"
TEST_OLD_BBOX_INPUT_PATH = r"/datasets/external/boss/test/bbox"
TEST_NEW_BBOX_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_8/test/new_bbox"
#TEST_NEW_BBOX_INPUT_PATH = r"/datasets/external/boss/test/new_bbox"
TEST_POSE_INPUT_PATH = r"/datasets/external/boss/test/pose"
TEST_POSE_AS_PICTURE_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_pose_16/test"
TEST_GAZE_INPUT_PATH = r"/datasets/external/boss/test/gaze"


## VAL PATHS
VAL_FRAME_INPUT_PATH =r"/datasets/external/boss/val/frame" 
PREPROCESSED_VAL_FRAME_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_128/val/frame"
VAL_POSE_AS_PICTURE_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_pose_16/val"
#PREPROCESSED_VAL_FRAME_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_256/val/frame"
VAL_OLD_BBOX_INPUT_PATH = r"/datasets/external/boss/val/bbox"
#VAL_NEW_BBOX_INPUT_PATH = r"/projects/arockiasamy/datasets/boss/presaved_8/val/new_bbox"
VAL_NEW_BBOX_INPUT_PATH = r"/datasets/external/boss/val/new_bbox"
VAL_POSE_INPUT_PATH = r"/datasets/external/boss/val/pose"
VAL_GAZE_INPUT_PATH = r"/datasets/external/boss/val/gaze"

## INPUT ADDITIONAL PARAMETERS

EXP_NO = "1" #type in the experiment episode number
INPUT_MODALITIES = 'bbox' #["bbox", "ocr", "pose"] #To visualize the input modality alone
BBOX_INPUT = 'new' #["old", "new"]
OCR_GRAPH = [[15, [10, 4], [17, 2]], [13, [16, 7], [18, 4]], [11, [16, 4], [7, 10]], [14, [10, 11], [7, 1]], [12, [10, 9], [16, 3]], [1, [7, 2], [9, 9], [10, 2]], [5, [8, 8], [6, 8]], [4, [9, 8], [7, 6]], [3, [10, 1], [8, 3], [7,4], [9,2], [6,1]], [2, [10, 1], [7, 7], [9,3]], [19, [10, 2], [26, 6]], 
                [20, [10, 7], [26, 5]], [22, [25, 4], [10, 8]], [23, [25, 15]], [21, [16, 5], [24, 8]]]
    
OLD_OCR_GRAPH = [[15, [10, 4], [17, 2]], [13, [16, 7], [18, 4]], [11, [16, 4], [7, 10]], [14, [10, 11], [7, 1]], [12, [10, 9], [16, 3]], [1, [
                7, 2], [9, 9]], [5, [8, 8], [6, 8]], [4, [9, 8], [7, 6]], [3, [10, 1], [8, 3]], [2, [10, 1], [7, 7]], [19, [10, 2], [26, 6]], 
                [20, [10, 7], [26, 5]], [22, [25, 4], [10, 8]], [24, [10, 11], [7, 1]], [23, [16, 4], [7, 10]]]
LIMBS_VIZ= [
    [0,1], [0,16], [0,15], 
    [16, 18], 
    [15,17], 
    [1,2], [2,3], [3,4],
    [1,5], [5,6], [6,7], 
    [1,8], 
    [8,12], [12,13], [13,14], [14,21], [14,19], [19,20],
    [8,9], [9,10], [10,11], [11,24], [11,22], [22,23]
        ]

POSE_PICTURE_LIMBS = [[0,1], [0,14], [0,13], [14, 16], [13,15], 
         [1,2], [2,3], [3,4],
         [1,5], [5,6], [6,7], 
         [1,8], 
         [8,11], [11,12],
         [8,9], [9,10]]
         
NO_NEED_POSE_JOINTS = [11, 14, 19, 20, 21, 22, 23, 24] #these joins are not visible in any of the frames throughout all the videos. So removing to reduce computation.
OLD_LIMBS = torch.tensor(
    [[17, 15, 15, 0, 0, 16, 16, 18, 0, 1, 4, 3, 3, 2, 2, 1, 1, 5, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 10, 11, 11, 24, 11, 23, 23, 22, 8, 12, 12, 13, 13, 14, 14, 21, 14, 19, 19, 20],
        [15, 17, 0, 15, 16, 0, 18, 16, 1, 0, 3, 4, 2, 3, 1, 2, 5, 1, 6, 5, 7, 6, 8, 1, 9, 8, 10, 9, 11, 10, 24, 11, 23, 11, 22, 23, 12, 8, 13, 12, 14, 13, 21, 14, 19, 14, 20, 19]],
    dtype=torch.long)

LIMBS = torch.tensor(
        [[17, 15, 15, 0, 0, 16, 16, 18, 0, 1, 4, 3, 3, 2, 2, 1, 1, 5, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 8, 12, 12, 13, 13, 14],
         [15, 17, 0, 15, 16, 0, 18, 16, 1, 0, 3, 4, 2, 3, 1, 2, 5, 1, 6, 5, 7, 6, 8, 1, 9, 8, 10, 9, 12, 8, 13, 12, 14, 13]],
        dtype=torch.long)
        
POSE_GRAPH_EDGE_INDEX = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 3, 5, 6, 8, 8, 9, 11, 13, 14], [1, 14, 13, 2, 5, 8, 3, 4, 6, 7, 9, 11, 10, 12, 15, 16]], dtype=torch.int32)

OBJECTS = ['none', 'apple', 'orange', 'lemon', 'potato', 'wine', 'wineopener',
                'knife', 'mug', 'peeler', 'bowl', 'chocolate', 'sugar', 'magazine',
                'cracker', 'chips', 'scissors', 'cap', 'marker', 'sardinecan', 'tomatocan',
                'plant', 'walnut', 'nail', 'waterspray', 'hammer', 'canopener']
    
N_JOINTS = 25

## INPUT 
POSE_AS_INPUT = 'coordinates' #coordinates, graph, picture
BBOX_AS_INPUT = 'new_coordinates' #old_coordinates, new_coordinates, picture
USE_OCR_AS = 'input' #'input' cost_func
SAVE_PATH = 'experiments/'

SA_LOAD_MODEL_PATH = 'best_model/Self_Attention_Model' #POSE_AS_INPUT = "coordinates", BBOX_AS_INPUT='new_coordinates', USE_OCR_AS = 'input' - Runtime error otherwise
CA_LOAD_MODEL_PATH = 'best_model/Hierarchical_Cross_Attention_Model' #POSE_AS_INPUT = "graph", BBOX_AS_INPUT='new_coordinates', USE_OCR_AS = 'input' - Runtime error otherwise

## Models Parameters
BATCH_SIZE = 32
EPOCHS = 1
LR = 0.0007
WEIGHT_DECAY = "Not_Used"
LAMBD = "Not_Used"
VGG_LAYERS_BBOX =  [16, "M", 16]
VGG_LAYERS_POSE = [16, "M", 8]
VGG_LAYERS = [64, "M", 64, "M", 128, "M", 128]
RESNET_LAYERS = [3,3,3,3]
MODEL_NUM_LAYERS = 4
MODEL_INPUT_DIM = 128
MODEL_NUM_HEAD = 4
MODEL_MLP_DIM = 256 #deals with FF after the MLA
MODEL_REPRESENTATION_DIM = 512 #deals with FF after the MLA #64 for cnn+lstm
MODEL_NUM_CLASSES = 27  
MODEL_ORIGINAL_INPUT_LENGTH_METHOD_1 = 128+16+16+16+16
MODEL_ORIGINAL_INPUT_LENGTH_METHOD_2 = 128+16+16+16+16 #if method2 is activated

##CROSS ATTENNTION
MODEL_ORIGINAL_INPUT_CA_LENGTH_METHOD_1 = 128+16+16


if __name__=='__main__':
    assert("This can not be run as a single file")