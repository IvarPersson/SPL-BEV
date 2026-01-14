import cv2
import torch
import torchinfo
from torch import nn
from torchvision import models
import numpy as np
from sskit.camera import grid2d
from torch.nn.functional import grid_sample

class ProjectAndSampleGrid(nn.Module):
    def __init__(self, voxel_grid_size_row, voxel_grid_size_col, voxel_grid_size_z, 
                 voxel_res, voxel_size, sampling_strategy, eps=1e-4, max_scale=1e3):
        super().__init__()
        self.eps = eps
        self.max_scale = max_scale
        self.voxel_grid_size_row = voxel_grid_size_row
        self.voxel_grid_size_col = voxel_grid_size_col
        self.voxel_grid_size_z = voxel_grid_size_z
        self.voxel_res = voxel_res
        self.voxel_size = voxel_size
        self.sampling_strategy = sampling_strategy

    def forward(self, feature_map, dist_poly, camera_matrix):
        """
        Create voxel grid points and project them to the image plane.
        """
        B, C, H, W = feature_map.shape
        z_vals = torch.linspace(0, self.voxel_grid_size_z, self.voxel_size[2]).to(camera_matrix)
        z_grid = z_vals.unsqueeze(0).expand(B, -1)
        # Batch_size x nbr z values x nbr features x height x width
        sampled = self.project_on_ground_batch(camera_matrix, dist_poly, feature_map, z_grid,
                                               width=self.voxel_grid_size_row,
                                               height=self.voxel_grid_size_col,
                                               resolution=self.voxel_res,
                                               center=(0, 0))
        sampled = torch.flip(sampled.transpose(3, 4), dims=[3, 4])
        #sampled = sampled.permute(0, 1, 2, 3, 4)
        return sampled
        # add your quant-safe guards (eps clamps, etc.)

    def project_on_ground_batch(self, camera_matrix, dist_poly, image, z_vals, width=70, height=120, resolution=10, center=(0,0)):
        B, C, H, W = image.shape
        Z = z_vals.shape[1]
        center_tensor = torch.tensor([width / 2, height / 2]).to(image)
        center = torch.as_tensor(center).to(image) - center_tensor  # (2,)
        # Generate grid (same for all samples)
        gnd = grid2d(width * resolution, height * resolution).to(image) / resolution + center  # (H'*W', 2)
        N = gnd.shape[0]
        M = gnd.shape[1]
        # Repeat across batch and Z
        gnd_expanded = gnd.unsqueeze(0).unsqueeze(0).expand(B, Z, N, M, 2)  # (B, Z, N, 2)
        # z_vals: (B, Z) → (B, Z, N, 1)
        z = z_vals.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, M, 1)  # (B, Z, N, 1)
        # pkt: (B, Z, N, 3)
        pkt = torch.cat([gnd_expanded, z], dim=-1)  # 3D points
        # Flatten batch for projection
        pkt_flat = pkt.view(B * Z, N * M, 3)
        cam_flat = camera_matrix.unsqueeze(1).expand(B, Z, 3, 4).reshape(B * Z, 3, 4)
        poly_flat = dist_poly.unsqueeze(1).expand(B, Z, -1).reshape(B * Z, -1)
        #IC = image.shape[1]
        img_flat = image.unsqueeze(1).expand(B, Z, C, H, W).reshape(B * Z, C, H, W)
        # Project: world_to_image must support batch (BZ, N, 3) → (BZ, N, 2)
        grid = self.world_to_image(cam_flat, poly_flat, pkt_flat)  # (BZ, N, 2)
        #world_to_image(cam_flat, dist_flat, pkt_flat)
        grid = grid.view(B * Z, int(height * resolution), int(width * resolution), 2)
        # Sample
        sampled = self.sample_image(img_flat, grid)  # (BZ, C, H', W')
        sampled = sampled.view(B, Z, C, sampled.shape[-2], sampled.shape[-1])
        return sampled  # (B, Z, C, H', W')

    def world_to_image(self, camera_matrix, distortion_poly, pkt):
        return self.distort(distortion_poly, self.world_to_undistorted(camera_matrix, pkt))
    
    def world_to_undistorted(self, camera_matrix, pkt):
        #camera_matrix = torch.as_tensor(camera_matrix)
        return self.to_cartesian(torch.matmul(self.to_homogeneous(pkt), camera_matrix.mT))

    def to_homogeneous(self, pkt):
        #pkt = torch.as_tensor(pkt)
        return torch.cat([pkt, torch.ones_like(pkt[..., 0:1])], -1)

    def to_cartesian(self, pkt):
        v = pkt[..., -1:]
        sign = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))   # sign(0) := +1
        abs_v = v.abs()
        # if |v| >= eps -> use v; else -> use sign*eps
        denom = torch.where(abs_v >= self.eps, v, sign * self.eps)
        tt = pkt[..., :-1]
        tmp = tt / denom
        tmp = torch.nan_to_num(tmp, 
                               posinf=torch.finfo(torch.float16).max, 
                               neginf=-torch.finfo(torch.float16).max)
        return tmp

    def distort(self, poly, pkt):
        rr = (pkt * pkt).sum(-1, keepdim=True).sqrt()                # >= 0
        rr_safe = torch.where(rr >= self.eps, rr, rr.new_full(rr.shape, self.eps))
        theta = torch.atan(rr_safe)                                  # consistent with denom
        rr2 = self.polyval(poly, theta)
        scale = rr2 / rr_safe
        tmp = pkt * scale
        return tmp

    def polyval(self, poly, pkt):
        sa = poly[..., 0:1]
        sa = pkt * sa.unsqueeze(-1) + poly[..., 1].unsqueeze(-1).unsqueeze(-1)
        sa = pkt * sa + poly[..., 2].unsqueeze(-1).unsqueeze(-1)
        sa = pkt * sa + poly[..., 3].unsqueeze(-1).unsqueeze(-1)
        sa = pkt * sa + poly[..., 4].unsqueeze(-1).unsqueeze(-1)
        sa = pkt * sa + poly[..., 5].unsqueeze(-1).unsqueeze(-1)
        sa = pkt * sa + poly[..., 6].unsqueeze(-1).unsqueeze(-1)
        sa = pkt * sa + poly[..., 7].unsqueeze(-1).unsqueeze(-1)
        sa = pkt * sa + poly[..., 8].unsqueeze(-1).unsqueeze(-1)
        # Lägg manuellt 9 i rad
        #for i in range(1, poly.shape[-1]):
        #    sa = pkt * sa + poly[..., i]
        return sa


    def sample_image(self, image, grid):
        #n, c, h, w = image.shape
        w = 3840
        h = 2160
        scaled_grid = 2 * w * grid
        #scaled_grid[..., 0] /= (w - 1)
        #scaled_grid[..., 1] /= (h - 1)
        scaled_grid_x = scaled_grid[..., 0] / float(w - 1)
        scaled_grid_y = scaled_grid[..., 1] / float(h - 1)
        scaled_grid = torch.stack([scaled_grid_x, scaled_grid_y], dim=-1)
        if image.dtype == torch.float16:
            image = image.float()
            scaled_grid = scaled_grid.float()
            output = grid_sample(image, scaled_grid, align_corners=False, mode=self.sampling_strategy)
            output = output.half()
            return output
        return grid_sample(image, scaled_grid, align_corners=False, mode=self.sampling_strategy)


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
        else:
            self.conv3 = nn.Identity()
        self.out_channels = out_channels
        self.act = nn.ReLU(inplace=True) if out_channels != 3 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.conv3(x)
        if getattr(residual, "dtype", None) == torch.quint8:
            q_scale = residual.q_scale()
            q_zero_point = residual.q_zero_point()
            residual_f = residual.dequantize()
            out_f = out.dequantize() if getattr(out, "dtype", None) == torch.quint8 else out
            summed = out_f + residual_f
            out = torch.quantize_per_tensor(summed, float(q_scale), int(q_zero_point), torch.quint8)
        else:
            out = out + residual
        out = self.act(out)
        return out


class BEVNetwork(nn.Module):
    def __init__(self,
                 res_block_channels:int=256,
                 debug=False, 
                 print_summary=False,
                 unet_features:list=None,
                 feature_extractor:str="", 
                 feature_channels:int=3,
                 res_net_tail=True,
                 voxel_grid_size=[100, 100, 3],
                 voxel_res:int=2,
                 sampling_strategy="bilinear",
                 tail_param:list=None):
        super(BEVNetwork, self).__init__()

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
        self.relu = nn.ReLU(inplace=True)
        self.sampling_strategy = sampling_strategy
        self.project_and_sample_voxel_grid = ProjectAndSampleGrid(self.voxel_grid_size_row, 
                                                                  self.voxel_grid_size_col, 
                                                                  self.voxel_grid_size_z,
                                                                  self.voxel_res,
                                                                  self.voxel_size,
                                                                  self.sampling_strategy)

        self.feature_extractor = feature_extractor

        if feature_extractor == "resnet50":
            self.model_name = "resnet-backbone"
            self.feat_extract = models.resnet50(pretrained=True)
            self.feat_extract = nn.Sequential(*list(self.feat_extract.children())[:-2])  # Cut before global average pooling
            self.nbr_features = feature_channels
            self.feat_compress = nn.Conv2d(2048, self.nbr_features, kernel_size=1, stride=1)
        elif feature_extractor == "unet":
            if unet_features is None:
                self.model_name = "unet-backbone"
            else:
                self.model_name = "unet-backbone-" + str(unet_features)
            self.feat_extract = UNet(in_channels=3, out_channels=feature_channels, features=unet_features)
            self.nbr_features = feature_channels
        elif feature_extractor == "mobile-unet":
            self.model_name = "mobile-unet-backbone" + str(unet_features)
            self.feat_extract = MobileUNet(in_channels=3, out_channels=feature_channels, features=unet_features)
            self.nbr_features = feature_channels
        elif feature_extractor == "":
            self.model_name = "no-backbone"
            self.nbr_features = 3
        else:
            raise ValueError(f"Feature extractor: {feature_extractor} not implemented") 

        # 3x3 convolutional layer for processing the BEV grid
        self.conv3x3 = nn.Conv2d(self.voxel_size[2] * self.nbr_features, self.nbr_features,
                                 kernel_size=3, stride=1, padding=1)
        self.debug = debug
        # Untrained ResNet-style blocks for further processing
        if self.res_net_tail == "my3dConv":
            self.res_blocks = nn.Sequential()
            ii = self.voxel_size[2] * self.nbr_features
            for p in tail_param:
                self.res_blocks.append(nn.Conv2d(in_channels=ii, out_channels=p, kernel_size=3, 
                                                 padding=1))
                ii = p
        if self.res_net_tail == "3dConv":
            self.res_blocks = nn.Sequential()
            ii = self.nbr_features
            for p in tail_param:
                self.res_blocks.append(nn.Conv3d(in_channels=ii, out_channels=p, 
                                                 kernel_size=1, padding=0))
                ii = p
            self.extra_block = nn.Conv2d(self.voxel_size[2] * ii, 3, kernel_size=3, stride=1, 
                                         padding=1)
        elif self.res_net_tail == True:
            self.res_block1 = ResNetBlock(self.nbr_features, self.res_block_channels)
            self.res_block2 = ResNetBlock(self.res_block_channels, 3)
        else:
            self.logreg = nn.Conv1d(self.nbr_features, 3, 1)

        self.input_quant = torch.ao.quantization.QuantStub()
        self.output_dequant = torch.ao.quantization.DeQuantStub()

        if print_summary:    
            virtual_batch = 2
            inp = [[virtual_batch, 3, 2160, 3840], 
                   [virtual_batch, 9],
                   [virtual_batch, 3, 4]]
            text = torchinfo.summary(self, inp, verbose=0)
            print(text)

    def forward(self, image_vals, dist_poly, camera_matrix):
        """
        Forward pass: projects the voxel grid and samples features for each point.
        """
        image_vals = self.input_quant(image_vals)
        if self.feature_extractor != "":
            features = self.feat_extract(image_vals)
            if hasattr(self, 'feat_compress'):
                features = self.feat_compress(features)
        else:
            features = image_vals
        assert features.shape[1] == self.nbr_features, \
                f"After feature extract - Expected {self.nbr_features} features, got {features.shape[1]}"
        if self.debug:
            sampled_features = self.project_and_sample_voxel_grid(image_vals, dist_poly, camera_matrix)
            cv2.imwrite("test.png", sampled_features[0,0,:,:,:].cpu().numpy().transpose(1,2,0)*255)
        sampled_features = self.project_and_sample_voxel_grid(features, dist_poly, camera_matrix)
        reshaped_features = sampled_features.reshape(dist_poly.shape[0], -1, self.voxel_size[1],
                                                     self.voxel_size[0])

        if self.res_net_tail == "my3dConv":
            bev_features = reshaped_features
            for block in self.res_blocks[:-1]:
                bev_features = block(bev_features)
                bev_features = self.relu(bev_features)
            bev_features = self.res_blocks[-1](bev_features)
        elif self.res_net_tail == "3dConv":
            bev_features = sampled_features.permute(0,2,1,3,4)
            for block in self.res_blocks:
                bev_features = block(bev_features)
                bev_features = self.relu(bev_features)
            bev_features = bev_features.permute(0,2,1,3,4)
            bev_features = bev_features.reshape(dist_poly.shape[0], -1, self.voxel_size[1], self.voxel_size[0])
            bev_features = self.extra_block(bev_features)
        else:
            bev_features = self.conv3x3(reshaped_features)
            if self.res_net_tail:
                bev_features = self.res_block1(bev_features)
                bev_features = self.res_block2(bev_features)
            else:
                bev_features = self.logreg(
                    bev_features.view(bev_features.shape[0],
                                  bev_features.shape[1],
                                  -1)).view(
                                      bev_features.shape[0],
                                      3,
                                      bev_features.shape[2],
                                      bev_features.shape[3])
        bev_features = self.output_dequant(bev_features)
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

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Decoder
        last_feat = features[-1] * 2
        for feature in reversed(features):
            self.upsample.append(nn.ConvTranspose2d(last_feat, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))
            last_feat = feature

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
            x = nn.functional.interpolate(x, size=skip_connections[idx].shape[2:], mode="nearest")
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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class MobileUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[8, 16, 32]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.upsample.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))

        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

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
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

