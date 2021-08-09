import torch.nn as nn
import torch


class SDS_VGG(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv3d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=1, stride=2))

        # block 2
        net.append(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=1, stride=2))

        # block 3
        net.append(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=1, stride=2))

        # block 4
        net.append(nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=1, stride=2))

        # block 5
        net.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=1, stride=2))

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512*1*2*2, out_features=1024))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=1024, out_features=512))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=512, out_features=self.num_classes))

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x1,x2):
        x1 = self.extract_feature(x1)
        x2 = self.extract_feature(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        feature = torch.abs(x1 - x2)
        classify_result = self.classifier(feature)
        return classify_result
