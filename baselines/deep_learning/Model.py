import torch.nn as nn
from torch.nn import functional as F

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Protonet(nn.Module):
    def __init__(self):
        super(Protonet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,128),
            conv_block(128,128),
            conv_block(128,128)
        )
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        return x.view(x.size(0),-1)

class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, (3,3), (2,2), (2,2))
    self.conv2 = nn.Conv2d(16, 32, (3,3), (2,2), (2,2))
    self.conv3 = nn.Conv2d(32, 64, (3,3), (2,2), (2,2))

    self.conv4 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
    self.conv5 = nn.Conv2d(32, 16, (3,3), (1,1), (1,1))
    
    self.convT1 = nn.ConvTranspose2d(64, 32, (3,3), (2,2), (2,2))
    self.convT2 = nn.ConvTranspose2d(32, 16, (3,3), (2,2), (2,2))
    self.convT3 = nn.ConvTranspose2d(16, 1, (3,3), (2,2), (2,2))

    self.dropout = nn.Dropout(0.5)
    self.bn1 = nn.BatchNorm2d(16)
    self.bn2 = nn.BatchNorm2d(32)
    self.bn3 = nn.BatchNorm2d(64)
    
    self.bn10 = nn.BatchNorm2d(1)
    self.bn11 = nn.BatchNorm2d(16)
    self.bn12 = nn.BatchNorm2d(32)
    
  def forward(self, x):
    x1 = self.conv1(x)
    x1 = F.leaky_relu(x1, 0.2)
    x1 = self.bn1(x1)
    x2 = self.conv2(x1)
    x2 = F.leaky_relu(x2, 0.2)
    x2 = self.bn2(x2)
    x3 = self.conv3(x2)
    x3 = F.leaky_relu(x3, 0.2)
    x3 = self.bn3(x3)

    
    d1 = self.convT1(x3, output_size= x2.shape[2:])
    d1 = self.conv4(torch.cat((d1, x2), dim=1))
    d1 = F.relu(d1)
    d1 = self.bn12(d1)
    d1 = self.dropout(d1)
    d2 = self.convT2(d1, output_size=x1.shape[2:])
    d2 = self.conv5(torch.cat((d2, x1), dim=1))
    d2 = F.relu(d2)
    d2 = self.bn11(d2)
    d2 = self.dropout(d2)
    d3 = self.convT3(d2, output_size=x.shape[2:])
    d3 = F.relu(d3)
    out1 = self.bn10(d3)
    
    return (out1, x3.view(x3.shape[0], -1))
