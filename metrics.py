import torch
from kornia.losses.ssim import SSIMLoss
#from lpips_pytorch import LPIPS,lpips

def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return (depth_pred - depth_gt).abs()

def acc_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.float()

# def lpips (image_pred, image_gt,net_type='alex'):
#     image_gt = image_gt.unsqueeze(0)
#     image_pred = image_pred.unsqueeze(0)
#     loss = lpips(image_pred,image_gt)

#     return loss

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    image_gt = image_gt.unsqueeze(0)
    image_pred = image_pred.unsqueeze(0)
    criterion = SSIMLoss(5)
    loss = criterion(image_gt, image_pred)
    return loss