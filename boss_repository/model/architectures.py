import torch
import torch.nn as nn
import logging
#from model.layers import ResNetBlock, BasicBlock, BBOX_ResNetBlock
#from configs.param import RESNET_LAYERS
from model.layers import VGGBlock, VGGBlockBbox, VGGBlockPose
from model.layers import SelfAttentionEncoder
from model.layers import CrossAttentionDecoder
from collections import OrderedDict
from torch_geometric.nn.conv import GATv2Conv
from configs.param import POSE_AS_INPUT, BATCH_SIZE, POSE_GRAPH_EDGE_INDEX, BBOX_AS_INPUT

## SELF ATTENTION VIT 
class SelfAttentionViT(nn.Module):
    def __init__(
        self,
        original_input_length,
        num_layers,
        num_heads,
        input_dim,
        mlp_dim,
        representation_dim,
        num_classes,
        device,
        is_ocr_cost_func,
        dropout=0.4,
        attention_dropout=0.4
    ):
        super(SelfAttentionViT, self).__init__()

        self.input_channel = 3

        self.pose_bbox_inputs_dim = 8

        self.device = device

        self.pose_as_input = POSE_AS_INPUT

        self.bbox_as_input = BBOX_AS_INPUT

        self.pose_graph_edge_index = POSE_GRAPH_EDGE_INDEX

        self.is_ocr_cost_func = is_ocr_cost_func
        
        if self.is_ocr_cost_func == True:
            logging.info("OCR as Cost Enabled")
            original_input_length = original_input_length - 16
            #print(original_input_length)

        self.vgg = VGGBlock(
            self.input_channel
        )

        self.left_pose_graph = GATv2Conv(in_channels = 3, out_channels= 8, n_heads=3)

        self.pose_left_avg_pool = nn.AdaptiveAvgPool2d((1, 8))

        self.left_pose_layers = VGGBlockPose()

        self.right_pose_layers = VGGBlockPose()

        # modality FFs
        self.left_pose_ff = nn.Sequential(
            nn.Linear(51, 8),
            nn.ReLU(inplace=True),
        )

        self.right_pose_graph = GATv2Conv(in_channels = 3, out_channels= 8, n_heads=3)

        self.pose_right_avg_pool = nn.AdaptiveAvgPool2d((1, 8))

        self.right_pose_ff = nn.Sequential(
            nn.Linear(51, 8),
            nn.ReLU(inplace=True),
        )
        self.left_gaze_ff = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
        )
        self.right_gaze_ff = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
        )

        self.bbox_layers = VGGBlockBbox()

        self.bbox_ff = nn.Sequential(
            nn.Linear(108, 16),
            nn.ReLU(inplace=True),
        )
        self.ocr_ff = nn.Sequential(
            nn.Linear(729, 16),
            nn.ReLU(inplace=True),
        )
        self.projection_mlp = nn.Linear(
            original_input_length, input_dim
        )
        self.encoder = SelfAttentionEncoder(
            num_layers, num_heads, input_dim, mlp_dim, dropout, attention_dropout
        )

        left_heads_layers = OrderedDict()
        left_heads_layers["pre_logits"] = nn.Linear(input_dim, representation_dim)
        left_heads_layers["act"] = nn.ReLU()
        left_heads_layers["head"] = nn.Linear(representation_dim, num_classes)

        right_heads_layers = OrderedDict()
        right_heads_layers["pre_logits"] = nn.Linear(input_dim, representation_dim)
        right_heads_layers["act"] = nn.ReLU()
        right_heads_layers["head"] = nn.Linear(representation_dim, num_classes)

        ocr_head_layers = OrderedDict()
        ocr_head_layers["head"] = nn.Linear(input_dim, 1)
        ocr_head_layers["act"] = nn.Sigmoid()

        self.left_beliefs_heads = nn.Sequential(left_heads_layers)
        self.right_beliefs_heads = nn.Sequential(right_heads_layers)
        self.ocr_head = nn.Sequential(ocr_head_layers)

    def forward(self, images, left_poses, right_poses,  left_gazes, right_gazes, bboxes, ocr_tensor):

        image_feat = self.vgg(images.to(self.device))
        
        if self.pose_as_input == "coordinates":
            left_poses_feat = self.left_pose_ff(left_poses.to(self.device))
            image_feat = torch.cat([image_feat, left_poses_feat], 1)
            #print("image + left pose")
            #print(image_feat.shape)

        elif self.pose_as_input == "graph":
            bs, l, c = left_poses.shape
            left_pose = self.left_pose_graph(left_poses.to(self.device).view(-1, c), self.pose_graph_edge_index.to(self.device))
            left_poses_feat = torch.squeeze(self.pose_left_avg_pool(left_pose.view(bs, l, self.pose_bbox_inputs_dim)))
            image_feat = torch.cat([image_feat, left_poses_feat.to(self.device)], 1)

        elif self.pose_as_input == "picture":
            left_poses_feat = self.left_pose_layers(left_poses.to(self.device))
            image_feat = torch.cat([image_feat, left_poses_feat], 1)

        if self.pose_as_input == "coordinates":
            right_poses_feat = self.right_pose_ff(right_poses.to(self.device))
            image_feat = torch.cat([image_feat, right_poses_feat], 1)
            #print("image + left and right pose")
            #print(image_feat.shape)

        elif self.pose_as_input == "graph":
            bs, l, c = right_poses.shape
            right_pose = self.right_pose_graph(right_poses.to(self.device).view(-1, c), self.pose_graph_edge_index.to(self.device))
            right_poses_feat = torch.squeeze(self.pose_right_avg_pool(right_pose.view(bs, l, self.pose_bbox_inputs_dim)))
            image_feat = torch.cat([image_feat, right_poses_feat.to(self.device)], 1)

        elif self.pose_as_input == "picture":
            logging.info("Right pose - {}".format(right_poses.shape))
            right_poses_feat = self.right_pose_layers(right_poses.to(self.device))
            image_feat = torch.cat([image_feat, right_poses_feat], 1)

        if left_gazes is not None:
            left_gazes_feat = self.left_gaze_ff(left_gazes.to(self.device))
            image_feat = torch.cat([image_feat, left_gazes_feat], 1)
            #print("image + left and right pose + left gaze")
            #print(image_feat.shape)
        
        if right_gazes is not None:
            right_gazes_feat = self.right_gaze_ff(right_gazes.to(self.device))
            image_feat = torch.cat([image_feat, right_gazes_feat], 1)
            #print("image + left and right pose + left gaze")
            #print(image_feat.shape)

        if self.bbox_as_input == "new_coordinates" or self.bbox_as_input == "old_coordinates":
            ## BBOX AS COORDINATES
            bboxes_feat = self.bbox_ff(bboxes.to(self.device))
            image_feat = torch.cat([image_feat, bboxes_feat], 1)

        elif self.bbox_as_input == "picture":
            bboxes_feat = self.bbox_layers(bboxes.to(self.device))
            image_feat = torch.cat([image_feat, bboxes_feat], 1)

        
        # if self.is_ocr_cost_func == False:
        if not self.is_ocr_cost_func == True:
            ocr_tensor_feat = self.ocr_ff(ocr_tensor.to(self.device))
            image_feat = torch.cat([image_feat, ocr_tensor_feat], 1)

        else:
            image_feat = image_feat


        proj_inp  = self.projection_mlp(image_feat)

        #logging.info("Proj Inp - {}".format(proj_inp))
        self_att_encoder = self.encoder(proj_inp)
        #logging.info("Tansformer Output - {}".format(self_att_encoder))

        left_beliefs = self.left_beliefs_heads(self_att_encoder)

        right_beliefs = self.right_beliefs_heads(self_att_encoder)
        
        if self.is_ocr_cost_func == True:
            logging.info("OCR as final layer")
            ocr_out = self.ocr_head(self_att_encoder)
        else:
            ocr_out = 0

        return left_beliefs, right_beliefs, ocr_out



### Cross ATTENTION VIT 
class CrossAttentionViT(nn.Module):
    def __init__(
        self,
        original_hc_input_length,
        original_oc_input_length,
        num_layers,
        num_heads,
        input_dim,
        mlp_dim,
        representation_dim,
        num_classes,
        device,
        is_ocr_cost_func,
        dropout=0.2,
        attention_dropout=0.1
    ):
        super(CrossAttentionViT, self).__init__()

        self.input_channel = 3
        
        self.pose_bbox_inputs_dim = 8

        self.pose_as_input = POSE_AS_INPUT

        self.bbox_as_input = BBOX_AS_INPUT

        self.pose_graph_edge_index = POSE_GRAPH_EDGE_INDEX

        self.is_ocr_cost_func = is_ocr_cost_func

        self.device = device

        if self.is_ocr_cost_func == True:
            logging.info("OCR as Cost Enabled")
            original_oc_input_length = original_oc_input_length - 16

        self.vgg = VGGBlock(
            self.input_channel
        )
        
        self.left_pose_graph = GATv2Conv(in_channels = 3, out_channels= 8, n_heads=3)

        self.pose_left_avg_pool = nn.AdaptiveAvgPool2d((1, 8))
        
        self.left_pose_layers = VGGBlockPose()

        self.right_pose_layers = VGGBlockPose()

        # modality FFs
        self.left_pose_ff = nn.Sequential(
            nn.Linear(51, 8),
            nn.ReLU(inplace=True),
        )

        self.right_pose_graph = GATv2Conv(in_channels = 3, out_channels= 8, n_heads=3)

        self.pose_right_avg_pool = nn.AdaptiveAvgPool2d((1, 8))

        self.right_pose_ff = nn.Sequential(
            nn.Linear(51, 8),
            nn.ReLU(inplace=True),
        )
        self.left_gaze_ff = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
        )
        self.right_gaze_ff = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
        )

        self.bbox_layers = VGGBlockBbox()

        self.bbox_ff = nn.Sequential(
            nn.Linear(108, 16),
            nn.ReLU(inplace=True),
        )
        self.ocr_ff = nn.Sequential(
            nn.Linear(729, 16),
            nn.ReLU(inplace=True),
        )
        self.projection_mlp_hc = nn.Linear(
            original_hc_input_length, input_dim
        )
        self.projection_mlp_oc = nn.Linear(
            original_oc_input_length, input_dim
        )
        self.encoder_hc = SelfAttentionEncoder(
            num_layers, num_heads, input_dim, mlp_dim, dropout, attention_dropout
        )
        self.encoder_oc = SelfAttentionEncoder(
            num_layers, num_heads, input_dim, mlp_dim, dropout, attention_dropout
        )
        self.decoder =  CrossAttentionDecoder(
            num_layers, num_heads, input_dim, mlp_dim, dropout, attention_dropout
        )
        left_heads_layers = OrderedDict()
        left_heads_layers["pre_logits"] = nn.Linear(input_dim, representation_dim)
        left_heads_layers["dropout"] = nn.Dropout(dropout)
        left_heads_layers["act"] = nn.ReLU()
        left_heads_layers["head"] = nn.Linear(representation_dim, num_classes)

        right_heads_layers = OrderedDict()
        right_heads_layers["pre_logits"] = nn.Linear(input_dim, representation_dim)
        right_heads_layers["dropout"] = nn.Dropout(dropout)
        right_heads_layers["act"] = nn.ReLU()
        right_heads_layers["head"] = nn.Linear(representation_dim, num_classes)

        ocr_head_layers = OrderedDict()
        ocr_head_layers["head"] = nn.Linear(input_dim, 1)
        ocr_head_layers["act"] = nn.Sigmoid()


        self.left_beliefs_heads = nn.Sequential(left_heads_layers)
        self.right_beliefs_heads = nn.Sequential(right_heads_layers)
        self.ocr_head = nn.Sequential(ocr_head_layers)

    def forward(self, images, left_poses, right_poses,  left_gazes, right_gazes, bboxes, ocr_tensor):
        #image_feat = self.resnet(images.to(self.device))
        image_feat_hc = self.vgg(images.to(self.device))
        image_feat_oc = self.vgg(images.to(self.device))

        if self.pose_as_input == "coordinates":
            left_poses_feat = self.left_pose_ff(left_poses.to(self.device))
            image_feat_hc = torch.cat([image_feat_hc, left_poses_feat], 1)

        elif self.pose_as_input == "graph":
            bs, l, c = left_poses.shape
            left_pose = self.left_pose_graph(left_poses.to(self.device).view(-1, c), self.pose_graph_edge_index.to(self.device))
            left_poses_feat = torch.squeeze(self.pose_left_avg_pool(left_pose.view(bs, l, self.pose_bbox_inputs_dim)))
            image_feat_hc = torch.cat([image_feat_hc, left_poses_feat.to(self.device)], 1)
        
        elif self.pose_as_input == "picture":
            left_poses_feat = self.left_pose_layers(left_poses.to(self.device))
            image_feat_hc = torch.cat([image_feat_hc, left_poses_feat], 1)

        if self.pose_as_input == "coordinates":
            right_poses_feat = self.right_pose_ff(right_poses.to(self.device))
            image_feat_hc = torch.cat([image_feat_hc, right_poses_feat], 1)

        elif self.pose_as_input == "graph":
            bs, l, c = right_poses.shape
            right_pose = self.right_pose_graph(right_poses.to(self.device).view(-1, c), self.pose_graph_edge_index.to(self.device))
            right_poses_feat = torch.squeeze(self.pose_right_avg_pool(right_pose.view(bs, l, self.pose_bbox_inputs_dim)))
            image_feat_hc = torch.cat([image_feat_hc, right_poses_feat.to(self.device)], 1)

        elif self.pose_as_input == "picture":
            #logging.info("Right pose - {}".format(right_poses.shape))
            right_poses_feat = self.right_pose_layers(right_poses.to(self.device))
            image_feat_hc = torch.cat([image_feat_hc, right_poses_feat], 1)

        if left_gazes is not None:
            left_gazes_feat = self.left_gaze_ff(left_gazes.to(self.device))
            image_feat_hc = torch.cat([image_feat_hc, left_gazes_feat], 1)
        
        if right_gazes is not None:
            right_gazes_feat = self.right_gaze_ff(right_gazes.to(self.device))
            image_feat_hc = torch.cat([image_feat_hc, right_gazes_feat], 1)

        if self.bbox_as_input == "new_coordinates" or self.bbox_as_input == "old_coordinates":
            #logging.info(bboxes.shape)
            ## BBOX AS COORDINATES
            bboxes_feat = self.bbox_ff(bboxes.to(self.device))
            image_feat_oc = torch.cat([image_feat_oc, bboxes_feat], 1)
        
        elif self.bbox_as_input == "picture":
            bboxes_feat = self.bbox_layers(bboxes.to(self.device))
            image_feat_oc = torch.cat([image_feat_oc, bboxes_feat], 1)

        if self.is_ocr_cost_func == False:
            #logging.info("OCR as Input Skipped")
            ocr_tensor_feat = self.ocr_ff(ocr_tensor.to(self.device))
            image_feat_oc = torch.cat([image_feat_oc, ocr_tensor_feat], 1)
        
        #PROJECTION
        # logging.info(image_feat_hc.shape)
        proj_inp_hc  = self.projection_mlp_hc(image_feat_hc)
        proj_inp_oc  = self.projection_mlp_oc(image_feat_oc)

        #SELF ATTENTION ENCODER
        # logging.info(proj_inp_hc.shape)
        self_att_encoder_hc = self.encoder_hc(proj_inp_hc)
        self_att_encoder_oc = self.encoder_oc(proj_inp_oc)
        
        #CROSS ATTENTION ENCODER
        cross_att_decoder = self.decoder(self_att_encoder_hc, self_att_encoder_oc)
        
        #MLP HEAD
        left_beliefs = self.left_beliefs_heads(cross_att_decoder)
        right_beliefs = self.right_beliefs_heads(cross_att_decoder)

        if self.is_ocr_cost_func == True:
            logging.info("OCR as final layer")
            ocr_out = self.ocr_head(cross_att_decoder)
        else:
            ocr_out = 0

        return left_beliefs, right_beliefs, ocr_out

if __name__=='__main__':
    assert("This can not be run as a single file")