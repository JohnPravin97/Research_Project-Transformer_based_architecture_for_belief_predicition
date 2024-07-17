import torch
import torch.nn as nn
from configs.param import VGG_LAYERS, VGG_LAYERS_BBOX, VGG_LAYERS_POSE
from collections import OrderedDict
import logging

class VGGBlock(nn.Module):
    def __init__(self, in_channels=3, 
                config=VGG_LAYERS):
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.config = config
        self.convolutional_layers = self.make_conv_layers(self.config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # function to create the convolutional layers as per the selected config
    def make_conv_layers(self, config):
        layers = []
        in_channels = self.in_channels
        for op in config:
            if op == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                layers += [
                           nn.Conv2d(in_channels=in_channels, 
                                      out_channels=op, kernel_size=2, 
                                      ),
                           nn.BatchNorm2d(op),
                           nn.ReLU()
                ]
                in_channels = op
        return nn.Sequential(*layers)
    
    # the forward pass
    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.avgpool(x) 
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        return x

class VGGBlockBbox(nn.Module):
    def __init__(self, in_channels=3, 
                config=VGG_LAYERS_BBOX):
        super(VGGBlockBbox, self).__init__()
        self.in_channels = in_channels
        self.config = config
        self.convolutional_layers = self.make_conv_layers(self.config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bbox_nn = nn.Linear(160, 16)
    
    # function to create the convolutional layers as per the selected config
    def make_conv_layers(self, config):
        layers = []
        in_channels = self.in_channels
        for op in config:
            if op == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                           nn.Conv2d(in_channels=in_channels, 
                                      out_channels=op, kernel_size=1),
                           nn.BatchNorm2d(op),
                           nn.ReLU()
                ]
                in_channels = op
        return nn.Sequential(*layers)
    
    # the forward pass
    def forward(self, bbox):
        batch_size, no_objects, channels, height, width = bbox.shape
        bbox_list = []
        for obj in range(no_objects):
            x = bbox[:,obj]
            #logging.info(x.shape)
            x = self.convolutional_layers(x)
            #logging.info(x.shape)
            x = self.avgpool(x)  
            #logging.info(x.shape)
            x = torch.flatten(x, 1)
            #logging.info(x.shape)
            bbox_list.append(x)
        #bbox_feats = torch.flatten(self.avgpool(torch.permute(torch.stack(bbox_list), (1,0,2,3,4))), 1)
        bbox_feats = torch.flatten(torch.permute(torch.stack(bbox_list), (1,0,2)), 1) # remove 1 X 1 grid and make vector of tensor shape 
        bbox_feats = self.bbox_nn(bbox_feats)
        return bbox_feats

class VGGBlockPose(nn.Module):
    def __init__(self, in_channels=17, 
                config=VGG_LAYERS_POSE):
        super(VGGBlockPose, self).__init__()
        self.in_channels = in_channels
        self.config = config
        self.convolutional_layers = self.make_conv_layers(self.config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # function to create the convolutional layers as per the selected config
    def make_conv_layers(self, config):
        layers = []
        in_channels = self.in_channels
        for op in config:
            if op == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                           nn.Conv2d(in_channels=in_channels, 
                                      out_channels=op, kernel_size=1),
                           nn.BatchNorm2d(op),
                           nn.ReLU()
                ]
                in_channels = op
        return nn.Sequential(*layers)
    
    # the forward pass
    def forward(self, pose):
        pose = pose.permute(0, 3, 1, 2)
        logging.info(pose.shape)
        x = self.convolutional_layers(pose)
        logging.info(x.shape)
        x = self.avgpool(x)  
        logging.info(x.shape)
        x = torch.flatten(x, 1)
        logging.info(x.shape)
        return x


## TRANSFORMER ENCODER BLOCKS and ENCODER BELOW
#https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
class SelfAttentionEncoderBlock(nn.Module):
    """Transformer encoder block. """
    def __init__(
            self, 
            num_heads,
            input_dim, 
            mlp_dim, 
            dropout, 
            attention_dropout
            ):
        super(SelfAttentionEncoderBlock, self).__init__()
        self.num_heads = num_heads
        
        self.ln_1 = nn.LayerNorm(input_dim)
        # Attention block
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        
        self.ln_2 = nn.LayerNorm(input_dim)
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, input_dim),
        )

    def forward(self, inp, mask=None):
        x = self.ln_1(inp)
        #print(x.shape)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        #print(x.shape)
        x = x + inp
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class SelfAttentionEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    def __init__(
        self,
        num_layers,
        num_heads,
        input_dim,
        mlp_dim,
        dropout=0, 
        attention_dropout=0
    ):
        super(SelfAttentionEncoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, input_dim).normal_(std=0.02))  # from BERT

        self.dropout = nn.Dropout(dropout)
        layers_dict = OrderedDict()
        for i in range(num_layers):
            layers_dict[f"encoder_layer_{i}"] = SelfAttentionEncoderBlock(
                num_heads,
                input_dim, 
                mlp_dim, 
                dropout,
                attention_dropout
            )
        self.layers = nn.Sequential(layers_dict)
        self.ln = nn.LayerNorm(input_dim)
    
    def forward(self, inputs):
        #torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, input_dim) got {input.shape}")
        inputs = inputs + self.pos_embedding
        return self.ln(self.layers(self.dropout(inputs)))


## CROSS ATTENTION TRANSFORMER ENCODER BLOCKS and ENCODER BELOW

class CrossAttentionDecoderBlock(nn.Module):
    """Transformer encoder block. """
    def __init__(
            self, 
            num_heads,
            input_dim, 
            mlp_dim, 
            dropout, 
            attention_dropout
            ):
        super(CrossAttentionDecoderBlock, self).__init__()
        self.num_heads = num_heads
        
        # Attention block
        self.cross_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        
        self.ln_2 = nn.LayerNorm(input_dim)
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, input_dim),
        )

    def forward(self, hc_inp, oc_inp):

        # Human Communication as Q and K
        x, _ = self.cross_attention(query = hc_inp, key = hc_inp, value = oc_inp, need_weights=False)

        # Object Context as Q and K
        # x, _ = self.cross_attention(query = oc_inp, key = oc_inp, value = hc_inp, need_weights=False)

        #print(x.shape)
        #x = self.dropout(x)
        x = x + oc_inp #value is only added
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class CrossAttentionDecoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    def __init__(
        self,
        num_layers,
        num_heads,
        input_dim,
        mlp_dim,
        dropout, 
        attention_dropout
    ):
        super(CrossAttentionDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        layers_dict = OrderedDict()
        self.layers = CrossAttentionDecoderBlock(
                num_heads,
                input_dim, 
                mlp_dim, 
                dropout,
                attention_dropout
            )
        self.ln = nn.LayerNorm(input_dim)
    
    def forward(self, HC_inputs, OC_inputs):
        #torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, input_dim) got {input.shape}")
        return self.ln(self.layers(hc_inp=self.dropout(HC_inputs), oc_inp=self.dropout(OC_inputs)))


if __name__=='__main__':
    assert("This can not be run as a single file")
    
