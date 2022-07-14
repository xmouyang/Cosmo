import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

class cnn_layers_1(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )

    def forward(self, x):
        x = self.features(x)

        return x


class cnn_layers_2(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            nn.Conv3d(input_size, 64, [5,5,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 64, [5,5,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 32, [5,5,1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(32, 16, [5,2,1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            )


    def forward(self, x):
        x = self.features(x)

        return x



class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.imu_cnn_layers = cnn_layers_1(input_size)
        self.skeleton_cnn_layers = cnn_layers_2(input_size)

    def forward(self, x1, x2):

        imu_output = self.imu_cnn_layers(x1)
        skeleton_output = self.skeleton_cnn_layers(x2)

        return imu_output, skeleton_output



class MyUTDModelFeature(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.encoder = Encoder(input_size)

        self.head_1 = nn.Sequential(

            nn.Linear(7552, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

        self.head_2 = nn.Sequential(

            nn.Linear(2688, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

    def forward(self, x1, x2):

        imu_output, skeleton_output = self.encoder(x1, x2)

        imu_output = F.normalize(self.head_1(imu_output.view(imu_output.size(0), -1)), dim=1)
        skeleton_output = F.normalize(self.head_2(skeleton_output.view(skeleton_output.size(0), -1)), dim=1)

        return imu_output, skeleton_output


"""
Attention block
Reference: https://github.com/philipperemy/keras-attention-mechanism
"""
class Attn(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduce_d1 = nn.Linear(7552, 1280)

        self.reduce_d2 = nn.Linear(2688, 1280)

        self.weight = nn.Sequential(

            nn.Linear(2560, 1280),
            nn.BatchNorm1d(1280),
            nn.Tanh(),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Linear(128, 2),
            nn.BatchNorm1d(2),
            nn.Tanh(),

            )

    def forward(self, hidden_state_1, hidden_state_2):

        new_1 = self.reduce_d1(hidden_state_1.view(hidden_state_1.size(0), -1))#hidden_state_1, [bsz, 16, 472]
        new_2 = self.reduce_d2(hidden_state_2.view(hidden_state_2.size(0), -1))#hidden_state_2, [bsz, 16, 168]

        concat_feature = torch.cat((new_1, new_2), dim=1) #[bsz, 1280*2]
        activation = self.weight(concat_feature)#[bsz, 2]

        score = F.softmax(activation, dim=1)

        attn_feature_1 = hidden_state_1 * (score[:, 0].view(-1, 1, 1)) 
        attn_feature_2 = hidden_state_2 * (score[:, 1].view(-1, 1, 1))

        fused_feature = torch.cat( (attn_feature_1, attn_feature_2), dim=2)

        return fused_feature, score[:, 0], score[:, 1]


class LinearClassifierAttn(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes):
        super(LinearClassifierAttn, self).__init__()

        self.attn = Attn()

        self.gru = nn.GRU(640, 120, 2, batch_first=True)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )

    def forward(self, feature1, feature2):

        feature1 = feature1.view(feature1.size(0), 16, -1)
        feature2 = feature2.view(feature2.size(0), 16, -1)

        fused_feature, weight1, weight2 = self.attn(feature1, feature2)

        fused_feature, _ = self.gru(fused_feature)
        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), -1)

        output = self.classifier(fused_feature)

        return output, weight1, weight2

