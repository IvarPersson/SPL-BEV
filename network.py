import cv2
import torch
import torchinfo
from torch import nn
from torchvision import models
import numpy as np
from sskit.camera import grid2d, world_to_image, sample_image


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if x.shape[1] != out.shape[1]:
            residual = self.conv3(x)
        out += residual
        if self.out_channels != 3:
            out = self.relu(out)
        return out

class SimpleBEVNetworkWithResNet(nn.Module):
    def __init__(self,
                 res_block_channels:int=256,
                 debug=False, 
                 print_summary=False,
                 unet_features:list=None,
                 feature_extractor:str="", 
                 feature_channels:int=3,
                 res_net_tail=True,
                 voxel_grid_size=[100, 100, 3],
                 voxel_res:int=2):
        super(SimpleBEVNetworkWithResNet, self).__init__()

        self.debug = debug
        self.voxel_grid_size_col = voxel_grid_size[1]
        self.voxel_grid_size_row = voxel_grid_size[0]
        self.voxel_grid_size_z = voxel_grid_size[2] # meters in z direction
        self.voxel_res = voxel_res # number of points per meter
        self.voxel_size = (self.voxel_grid_size_col*self.voxel_res,
                           self.voxel_grid_size_row*self.voxel_res,
                           self.voxel_grid_size_z*self.voxel_res)
        self.res_block_channels = res_block_channels

        self.res_net_tail = res_net_tail
        self.feat_extractor_name = feature_extractor
        self.nbr_features = feature_channels

        if self.feat_extractor_name == "resnet50":
            self.model_name = "resnet-backbone"
            self.feat_extract = models.resnet50(pretrained=True)
            self.feat_extract = nn.Sequential(*list(self.feat_extract.children())[:-2])
            self.feat_compress = nn.Conv2d(2048, self.nbr_features, kernel_size=1, stride=1)
        elif self.feat_extractor_name == "unet":
            assert unet_features is not None, "UNet features must be specified"
            self.model_name = "unet-backbone-" + str(unet_features)
            self.feat_extract = UNet(in_channels=3, out_channels=self.nbr_features,
                                     features=unet_features)
        elif self.feat_extractor_name == "":
            self.model_name = "no-backbone"
            self.nbr_features = 3
        else:
            raise ValueError(f"Feature extractor: {self.feat_extractor_name} not implemented")        

        # 3x3 convolutional layer for processing the BEV grid
        self.conv3x3 = nn.Conv2d(self.voxel_size[2] * self.nbr_features, self.nbr_features,
                                 kernel_size=3, stride=1, padding=1)

        # Untrained ResNet-style blocks for further processing
        if self.res_net_tail:
            self.BEVest = [ResNetBlock(self.nbr_features, self.res_block_channels), 
                           ResNetBlock(self.res_block_channels, 3)]
        else:
            self.BEVest = nn.Conv1d(self.nbr_features, 3, 1)

        if print_summary:    
            virtual_batch = 2
            inp = [[virtual_batch, 3, 2160, 3840], 
                   [virtual_batch, 9],
                   [virtual_batch, 3, 4]]
            text = torchinfo.summary(self, inp, verbose=0)
            print(text)


    def project_and_sample_voxel_grid(self, dist_poly, camera_matrix, feature_map):
        """
        Create voxel grid points and project them to the image plane.
        """
        z_vals = np.linspace(0, self.voxel_grid_size_z, self.voxel_size[2])
        bev_features = torch.zeros((dist_poly.shape[0], len(z_vals), feature_map.shape[1],
                                   self.voxel_grid_size_row*self.voxel_res,
                                   self.voxel_grid_size_col*self.voxel_res)).to(feature_map.device)
        for batch in range(dist_poly.shape[0]):
            for ii, z in enumerate(z_vals):
                aa = self.project_on_ground(camera_matrix[batch:batch+1, :, :],
                                       dist_poly[batch:batch+1, :],
                                       feature_map[batch:batch+1, :, :, :],
                                       width=self.voxel_grid_size_row,
                                       height=self.voxel_grid_size_col,
                                       resolution=self.voxel_res, 
                                       center=(0,0), z=z)
                tmp = torch.transpose(aa[0], 2, 1)
                tmp = torch.flip(tmp, (1,2))
                bev_features[batch, ii, :, :, :] = tmp
        return bev_features
    
    def project_on_ground(self, camera_matrix, dist_poly, image, width=70, height=120,
                          resolution=10, center=(0,0), z=0):
        center = torch.as_tensor(center, device=image.device) - torch.tensor([width/2, height/2],
                                                                             device=image.device)
        gnd = grid2d(width * resolution, height * resolution).to(image.device) / resolution + center
        pkt = gnd.reshape(-1, 2)
        pkt = torch.cat([pkt, z * torch.ones_like(pkt[..., 0:1])], -1)
        grid = world_to_image(camera_matrix, dist_poly, pkt).reshape(gnd.shape)
        return sample_image(image, grid[None])


    def forward(self, image_vals, dist_poly, camera_matrix):
        """
        Forward pass: projects the voxel grid and samples features for each point.
        """
        if self.feat_extractor_name != "":
            features = self.feat_extract(image_vals)
            if hasattr(self, 'feat_compress'):
                features = self.feat_compress(features)
        else:
            features = image_vals
        assert features.shape[1] == self.nbr_features, \
                f"After feature extract - Expected {self.nbr_features} features, got {features.shape[1]}"
        if self.debug:
            sampled_features = self.project_and_sample_voxel_grid(dist_poly, camera_matrix,
                                                                  image_vals)
            cv2.imwrite("test.png", sampled_features[0,0,:,:,:].cpu().numpy().transpose(1,2,0)*255)
        sampled_features = self.project_and_sample_voxel_grid(dist_poly, camera_matrix, features)
        reshaped_features = sampled_features.reshape(dist_poly.shape[0], -1, self.voxel_size[1],
                                                     self.voxel_size[0])

        # Apply 3x3 convolution on the restructured feature map
        bev_features = self.conv3x3(reshaped_features)

        # Pass through two ResNet-style blocks
        if self.res_net_tail:
            bev_features = self.BEVest[0](bev_features)
            bev_features = self.BEVest[1](bev_features)
        else:
            end_shape = bev_features.shape
            bev_features = self.BEVest(
                bev_features.view(bev_features.shape[0], bev_features.shape[1], -1))
            bev_features = bev_features.view(end_shape[0], 3, end_shape[2], end_shape[3])
        return bev_features 


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[8, 16, 32]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.upsample.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))
        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx, (up, dec) in enumerate(zip(self.upsample, self.decoder)):
            x = up(x)
            if x.shape != skip_connections[idx].shape:
                x = nn.functional.interpolate(x, size=skip_connections[idx].shape[2:])
            x = torch.cat((skip_connections[idx], x), dim=1)
            x = dec(x)
        return self.final_conv(x)

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )