import torch
import torch.nn as nn
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
        # x = x.view(x.size(0), 16, -1)

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
        # x = x.view(x.size(0), 16, -1)

        return x



class MyUTDmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.imu_cnn_layers = cnn_layers_1(input_size)
        self.skeleton_cnn_layers = cnn_layers_2(input_size)

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

    def forward(self, x1, x2):

        imu_output = self.imu_cnn_layers(x1)
        skeleton_output = self.skeleton_cnn_layers(x2)

        # fused_feature = (imu_output + skeleton_output) / 2 # weighted sum

        imu_output = imu_output.view(imu_output.size(0), 16, -1)
        skeleton_output = skeleton_output.view(skeleton_output.size(0), 16, -1)

        fused_feature = torch.cat((imu_output,skeleton_output), dim=2) #concate

        fused_feature, _ = self.gru(fused_feature)
        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), -1)

        output = self.classifier(fused_feature)

        return output

