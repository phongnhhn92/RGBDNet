from .modules import *
from .unet_model import UNet
import torchvision.transforms as T
from torchvision.utils import save_image


class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """

    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act),
        )

        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
        )

        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
        )

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x)  # (B, 8, H, W)
        conv1 = self.conv1(conv0)  # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1)  # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        feats = {"level_0": feat0, "level_1": feat1, "level_2": feat2}

        return feats


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(32),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(16),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(8),
        )

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x


class NovelDepthNet(nn.Module):
    def __init__(self,
                 n_depths=[8, 32, 48],
                 interval_ratios=[1, 2, 4],
                 norm_act=InPlaceABN,
                 opts=None
                 ):
        super(NovelDepthNet, self).__init__()
        self.opts = opts
        self.levels = 3  # 3 depth levels
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.feature_net = FeatureNet(norm_act)
        for l in range(self.levels):
            cost_reg_l = CostRegNet(8 * 2 ** l, norm_act)
            setattr(self, f"cost_reg_{l}", cost_reg_l)

        # for l in range(self.levels):
        #     unet_l = UNet(n_channels=3, out_channels=3)
        #     setattr(self, f"unet_{l}", unet_l)
        # Single Unet generator
        self.unet = UNet(n_channels=3, out_channels=3)

        self.unpreprocess = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

    def predict_depth(self, feats, proj_mats, depth_values, cost_reg):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V-1, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        src_feats = feats.permute(1, 0, 2, 3, 4)
        proj_mats = proj_mats.permute(1, 0, 2, 3)

        warp_volumes = []
        for src_feat, proj_mat in zip(src_feats, proj_mats):
            dict_ = homo_warp(src_feat, proj_mat, depth_values)
            warped_volume = dict_['warped_feats']
            warped_volume = warped_volume.to(feats.dtype)
            warp_volumes.append(warped_volume)
        warp_volumes = torch.stack([warp_volumes[i] for i in range(len(warp_volumes))], dim=1)

        PSV = torch.mean(warp_volumes, dim=1)
        cost_reg_ = cost_reg(PSV).squeeze(1)
        prob_volume = F.softmax(cost_reg_, 1)  # (B, D, h, w)
        del cost_reg_, PSV
        depth = depth_regression(prob_volume, depth_values)

        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(
                F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1),
                stride=1,
            ).squeeze(
                1
            )  # (B, D, h, w)
            # find the (rounded) index that is the final prediction
            depth_index = depth_regression(
                prob_volume,
                torch.arange(D, device=prob_volume.device, dtype=prob_volume.dtype),
            ).long()  # (B, h, w)
            depth_index = torch.clamp(depth_index, 0, D - 1)
            # the confidence is the 4-sum probability at this index
            confidence = torch.gather(
                prob_volume_sum4, 1, depth_index.unsqueeze(1)
            ).squeeze(
                1
            )  # (B, h, w)

        return depth, confidence

    def warp2NovelView(self, pred_depth, source_imgs, proj_mats, level):
        V = source_imgs.shape[1]
        list_warp = []
        list_weights = []
        list_input_depths = []
        for i in range(V):
            source_img = source_imgs[:, i]
            source_img_resized = F.interpolate(
                source_img,
                scale_factor=1 / 2 ** level,
                mode="bilinear",
                align_corners=True,
            )  # (B, 3, h, w)
            proj_mat = proj_mats[:, i]
            depth_values = pred_depth.unsqueeze(1)
            dict_ = homo_warp(source_img_resized, proj_mat, depth_values,
                                                           return_weights=True)
            warp_novel_view = dict_['warped_feats']
            inverse_depth_src = dict_['inv_depth']
            warp_novel_view = warp_novel_view.squeeze(2)
            list_input_depths.append(1.0 / inverse_depth_src.squeeze(1))
            list_warp.append(warp_novel_view)
            list_weights.append(inverse_depth_src)

        weights_sum = 0
        for w in list_weights:
            weights_sum += w
        list_weights_norm = []
        for w in list_weights:
            weights = w / weights_sum
            list_weights_norm.append(weights)

        combined_warps = 0
        for i in range(V):
            combined_warps += list_weights_norm[i] * list_warp[i]

        return combined_warps, list_input_depths

    def forward(self, imgs, proj_mats, proj_mats_ref2inputs, init_depth_min, depth_interval):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V-1, self.levels, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # First view is the reference image (novel image)
        # The rest are input views

        source_imgs = imgs[:, 1:]

        B, V, _, H, W = source_imgs.shape
        results = {}

        source_imgs = source_imgs.reshape(B * V, 3, H, W)
        source_feats = self.feature_net(
            source_imgs
        )  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        source_imgs = source_imgs.reshape(B, V, 3, H, W)
        for l in reversed(range(self.levels)):
            proj_mats_l = proj_mats[:, :, l]
            depth_interval_l = depth_interval * self.interval_ratios[l]
            feats_l = source_feats[f"level_{l}"]  # (B*V, C, h, w)
            feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)

            D = self.n_depths[l]
            if l == self.levels - 1:
                h, w = feats_l.shape[-2:]
                if isinstance(init_depth_min, float):
                    depth_values = init_depth_min + depth_interval_l * torch.arange(
                        0, D, device=imgs.device, dtype=imgs.dtype
                    )  # (D)
                    depth_values = depth_values.reshape(1, D, 1, 1).repeat(B, 1, h, w)
                else:
                    depth_values = init_depth_min.unsqueeze(
                        1
                    ) + depth_interval_l.unsqueeze(1) * torch.arange(
                        0, D, device=imgs.device, dtype=imgs.dtype
                    ).unsqueeze(
                        0
                    )  # (B, D)
                    depth_values = depth_values.reshape(B, D, 1, 1).repeat(1, 1, h, w)
            else:
                depth_l_1 = depth_l.detach()  # the depth of previous level
                depth_l_1 = F.interpolate(
                    depth_l_1.unsqueeze(1),
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )  # (B, 1, h, w)
                depth_values = get_depth_values(depth_l_1, D, depth_interval_l)
                del depth_l_1

            depth_l, confidence_l = self.predict_depth(
                feats_l, proj_mats_l, depth_values, getattr(self, f"cost_reg_{l}")
            )

            if self.opts.loss_type == 'unsup' or self.opts.loss_type == 'sup':
                novel_view_warps, inputDepth_list_l = self.warp2NovelView(depth_l, source_imgs, proj_mats_l, l)

                if l == self.levels - 1:
                    final_view_l = self.unet(novel_view_warps)

                    if self.opts.use_consistentLoss:
                        # Add cycle consistency reconstruction
                        reconstructed_input_l = []
                        for k in range(len(inputDepth_list_l)):
                            reconstructed_input_l_k = homo_warp(final_view_l, proj_mats_ref2inputs[:, k, l],
                                                                inputDepth_list_l[k].unsqueeze(1), False)['warped_feats'].squeeze(2)
                            # Use generator network takes too much memory
                            #reconstructed_input_l_k = self.unet(reconstructed_input_l_k)
                            reconstructed_input_l.append(reconstructed_input_l_k)

                        reconstructed_input_previous_l = torch.stack(reconstructed_input_l)
                        results[f"reconstructed_input_{l}"] = reconstructed_input_l
                else:
                    final_view_rs = F.interpolate(
                        final_view_l,
                        scale_factor=2.0,
                        mode="bilinear",
                        align_corners=True,
                    )  # (B, 3, h, w)
                    novel_view_warps += final_view_rs
                    novel_view_warps = novel_view_warps.div(2)
                    final_view_l = self.unet(novel_view_warps)

                    if self.opts.use_consistentLoss:
                        # Add cycle consistency reconstruction
                        reconstructed_input_l = []
                        for k in range(len(inputDepth_list_l)):
                            reconstructed_input_l_k = homo_warp(final_view_l, proj_mats_ref2inputs[:, k, l],
                                                                inputDepth_list_l[k].unsqueeze(1), False)['warped_feats'].squeeze(2)
                            final_reconView_rs = F.interpolate(
                                reconstructed_input_previous_l[k],
                                scale_factor=2.0,
                                mode="bilinear",
                                align_corners=True)
                            reconstructed_input_l_k += final_reconView_rs
                            reconstructed_input_l_k = reconstructed_input_l_k.div(2)
                            #reconstructed_input_l_k = self.unet(reconstructed_input_l_k)
                            reconstructed_input_l.append(reconstructed_input_l_k)
                        reconstructed_input_previous_l = torch.stack(reconstructed_input_l)
                        results[f"reconstructed_input_{l}"] = reconstructed_input_l
                del reconstructed_input_l,reconstructed_input_l_k
                results[f"warp_view_{l}"] = final_view_l
                results[f"input_depths_{l}"] = inputDepth_list_l

            del feats_l, proj_mats_l, depth_values
            results[f"depth_{l}"] = depth_l
            results[f"confidence_{l}"] = confidence_l

        return results
