import torch, torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, targets, masks):
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2 ** (1 - l)
        return loss


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))

        return 1 - torch.mean(prod / (fake_norm * real_norm))


class Depth_Loss(nn.Module):

    def __init__(self, levels=3):
        super(Depth_Loss, self).__init__()
        self.levels = levels
        self.normal_Loss = NormalLoss()
        self.grad_Loss = GradLoss()
        self.L1loss = nn.SmoothL1Loss(reduction='mean')

    def imgrad(self, img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)

        return grad_y, grad_x

    def imgrad_yx(self, img):
        N, C, _, _ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)

    def forward(self, inputs, targets, masks):
        loss = 0
        grad_factor = 1.0
        normal_factor = 1.0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']

            # Get grad
            grad_real, grad_fake = self.imgrad_yx(depth_gt_l.unsqueeze(1)), self.imgrad_yx(depth_pred_l.unsqueeze(1))

            # Gradient loss
            grad_loss = self.grad_Loss(grad_fake, grad_real) * grad_factor
            normal_loss = self.normal_Loss(grad_fake, grad_real) * normal_factor

            # Log l1 loss
            L1_loss = self.L1loss(depth_pred_l[mask_l], depth_gt_l[mask_l])

            loss += (grad_loss + normal_loss + L1_loss) * 2 ** (1 - l)
        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        # No need to normalize target because we have done it in the preprocessing step
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def calc_align_loss(self, gen, tar):
        def sum_u_v(x):
            area = x.shape[-2] * x.shape[-1]
            return torch.sum(x.view(-1, area), -1) + 1e-7

        device = tar.device
        coord_y, coord_x = torch.meshgrid(torch.arange(-1, 1, 1 / 14, device=device),
                                          torch.arange(-1, 1, 1 / 14, device=device))

        sum_gen = sum_u_v(gen)
        sum_tar = sum_u_v(tar)
        c_u_k = sum_u_v(coord_x * tar) / sum_tar
        c_v_k = sum_u_v(coord_y * tar) / sum_tar
        c_u_k_p = sum_u_v(coord_x * gen) / sum_gen
        c_v_k_p = sum_u_v(coord_y * gen) / sum_gen
        out = F.mse_loss(torch.stack([c_u_k, c_v_k], -1), torch.stack([c_u_k_p, c_v_k_p], -1), reduction='mean')
        return out

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        feats = x.view(b * c, h * w)
        g = torch.mm(feats, feats.t())
        return g.div(b * c * h * w)

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target

        average_err_map = torch.mean(torch.pow(x - y, 2), dim=1)
        min_map = torch.min(average_err_map).repeat(*average_err_map.shape)
        max_map = torch.max(average_err_map).repeat(*average_err_map.shape)
        guidance_map = (average_err_map - min_map) / (max_map - min_map)
        guidance_map = guidance_map.unsqueeze(1)

        for i, block in enumerate(self.blocks):
            if i != 0:
                avg_pool = torch.nn.AvgPool2d(2, stride=2)
                guidance_map = avg_pool(guidance_map)
            b, c, w, h = x.shape
            weight = 1e3 / (w * h)

            x = block(x)
            y = block(y)

            # Guidance loss
            if i == 1 or i == 2:
                loss += weight * torch.nn.functional.mse_loss(x * guidance_map, y * guidance_map, reduction='mean')

            # Alignment loss
            if i == 3:
                loss += self.calc_align_loss(x, y)

            # Style loss
            if i == 1 or i == 2 or i == 3:
                loss += torch.nn.functional.mse_loss(self.gram_matrix(x), self.gram_matrix(y), reduction='mean')

            # Feature loss
            loss += torch.nn.functional.mse_loss(x, y, reduction='mean')
        return loss


class UnSupervised_SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(UnSupervised_SL1Loss, self).__init__()
        self.levels = levels
        self.L1loss = nn.SmoothL1Loss(reduction='mean')
        self.vggLoss = VGGPerceptualLoss()

        self.alpha = 1.0
        self.beta = 1.0

    def forward(self, results, imgs, targets, masks, use_consistentLoss=False):
        ref_img = imgs[:, 0]
        source_imgs = imgs[:, 1:]
        image_loss = 0
        # SL1 loss with GT view
        for l in reversed(range(self.levels)):
            ref_img_resized = F.interpolate(
                ref_img,
                scale_factor=1 / 2 ** l,
                mode="bilinear",
                align_corners=True,
            )  # (B, 3, h, w)

            pred_view = results[f"warp_view_{l}"]
            image_loss += self.alpha * self.L1loss(ref_img_resized, pred_view) * 2 ** (1 - l)
            image_loss += self.beta * self.vggLoss(ref_img_resized, pred_view) * 2 ** (1 - l)

            if use_consistentLoss:
                for k in range(source_imgs.shape[1]):
                    source_img_k = source_imgs[:, k]
                    source_img_k_resized = F.interpolate(
                        source_img_k,
                        scale_factor=1 / 2 ** l,
                        mode="bilinear",
                        align_corners=True,
                    )  # (B, 3, h, w)

                    recon_img = results[f"reconstructed_input_{l}"][k]
                    image_loss += self.alpha * self.L1loss(source_img_k_resized, recon_img) * 2 ** (1 - l)
                    #image_loss += self.beta * self.vggLoss(source_img_k_resized, recon_img) * 2 ** (1 - l)

        return image_loss


class Supervised_SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(Supervised_SL1Loss, self).__init__()
        self.levels = levels
        self.L1loss = nn.SmoothL1Loss(reduction='mean')
        self.vggLoss = VGGPerceptualLoss()
        self.depthLoss = Depth_Loss(self.levels)

        self.alpha = 1.0
        self.beta = 1.0

    def forward(self, results, imgs, targets, masks, use_consistentLoss=False):
        ref_img = imgs[:, 0]
        source_imgs = imgs[:, 1:]
        depth_loss = self.depthLoss(results, targets, masks)
        image_loss = 0
        # SL1 loss with GT view
        for l in reversed(range(self.levels)):
            ref_img_resized = F.interpolate(
                ref_img,
                scale_factor=1 / 2 ** l,
                mode="bilinear",
                align_corners=True,
            )  # (B, 3, h, w)

            pred_view = results[f"warp_view_{l}"]
            image_loss += self.alpha * self.L1loss(ref_img_resized, pred_view) * 2 ** (1 - l)
            image_loss += self.beta * self.vggLoss(ref_img_resized, pred_view) * 2 ** (1 - l)

            if use_consistentLoss:
                for k in range(source_imgs.shape[1]):
                    source_img_k = source_imgs[:, k]
                    source_img_k_resized = F.interpolate(
                        source_img_k,
                        scale_factor=1 / 2 ** l,
                        mode="bilinear",
                        align_corners=True,
                    )  # (B, 3, h, w)

                    recon_img = results[f"reconstructed_input_{l}"][k]
                    image_loss += self.alpha * self.L1loss(source_img_k_resized, recon_img) * 2 ** (1 - l)
                    #image_loss += self.beta * self.vggLoss(source_img_k_resized, recon_img) * 2 ** (1 - l)

        return depth_loss + image_loss


loss_dict = {'unsup': UnSupervised_SL1Loss, 'sup': Supervised_SL1Loss}
