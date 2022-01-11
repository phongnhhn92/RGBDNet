import os

from inplace_abn import InPlaceABN
from pytorch_lightning import LightningModule, Trainer
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint,TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
# Dataset
from torch.utils.data import DataLoader

from datasets import dataset_dict
# Loss function
from losses import loss_dict
# Metrics
from metrics import *
# Models
from models.NovelDepthNet import NovelDepthNet
from opt import get_opts
# Optimizer, Scheduler, Visualization
from utils import *


# Hello
class NovelDepthSystem(LightningModule):
    def __init__(self, opts):
        super(NovelDepthSystem, self).__init__()
        self.opts = opts
        # to unnormalize image for visualization
        # self.unpreprocess = T.Normalize(
        #     mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        #     std=[1 / 0.5, 1 / 0.5, 1 / 0.5],
        # )
        self.unpreprocess = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        self.loss = loss_dict[opts.loss_type](opts.levels)
        self.model = NovelDepthNet(n_depths=self.opts.n_depths,
                                   interval_ratios=self.opts.interval_ratios,
                                   norm_act=InPlaceABN,
                                   opts=self.opts)
        # load model if checkpoint path is provided
        if self.opts.ckpt_path != "":
            print("Load model from", self.opts.ckpt_path)
            load_ckpt(
                self.model, self.opts.ckpt_path, self.opts.prefixes_to_ignore
            )

    def decode_batch(self, batch):
        imgs = batch["imgs"]
        proj_mats = batch["proj_mats"]
        depths = batch["depths"]
        masks = batch["masks"]
        init_depth_min = batch["init_depth_min"]
        depth_interval = batch["depth_interval"]
        proj_mats_ref2inputs = batch["proj_mats_ref2inputs"]
        return imgs, proj_mats, depths, masks, init_depth_min, depth_interval, proj_mats_ref2inputs

    def forward(self, imgs, proj_mats, proj_mats_ref2inputs, init_depth_min, depth_interval):
        return self.model(imgs, proj_mats, proj_mats_ref2inputs, init_depth_min, depth_interval)

    def setup(self, stage):
        dataset = dataset_dict[self.opts.dataset_name]
        self.train_dataset = dataset(
            root_dir=self.opts.root_dir,
            split="train",
            n_views=self.opts.n_views,
            levels=self.opts.levels,
            depth_interval=self.opts.depth_interval,
        )
        self.val_dataset = dataset(
            root_dir=self.opts.root_dir,
            split="val",
            n_views=self.opts.n_views,
            levels=self.opts.levels,
            depth_interval=self.opts.depth_interval,
        )

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.opts, self.model)
        scheduler = get_scheduler(self.opts, self.optimizer)

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=self.opts.batch_size,
            pin_memory=True,
        )

    def getLoss(self, loss_type, results, imgs, depths, masks):        
        return self.loss(results, imgs, depths, masks, self.opts.use_consistentLoss)
        

    def training_step(self, batch, batch_nb):
        (
            imgs,
            proj_mats,
            depths,
            masks,
            init_depth_min,
            depth_interval,
            proj_mats_ref2inputs
        ) = self.decode_batch(batch)
        results = self(imgs, proj_mats, proj_mats_ref2inputs, init_depth_min, depth_interval)

        loss = self.getLoss(opts.loss_type, results, imgs, depths, masks)
        sync_log = True if self.opts.num_gpus > 1 else False
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=sync_log)
        with torch.no_grad():
            if batch_nb % 1000 == 0:
                img_ = self.unpreprocess(imgs[0, 0]).cpu()  # batch 0, ref image
                depth_gt_ = visualize_depth(depths["level_0"][0])
                depth_pred_ = visualize_depth(
                    results["depth_0"][0] * masks["level_0"][0]
                )
                prob = visualize_prob(results["confidence_0"][0] * masks["level_0"][0])
                stack = torch.stack(
                    [img_, depth_gt_, depth_pred_, prob]
                )  # (4, 3, H, W)
                self.logger.experiment.add_images(
                    "train/image_GT_pred_prob", stack, self.global_step
                )

                list_img = []
                for i in range(self.opts.n_views):
                    img_i = self.unpreprocess(imgs[0, i]).cpu()
                    list_img.append(img_i)
                stack_imgs = torch.stack([list_img[i] for i in range(self.opts.n_views)])
                self.logger.experiment.add_images(
                    "train/GT_inputviews", stack_imgs, self.global_step
                )

                final_warp = self.unpreprocess(results["warp_view_0"][0]).unsqueeze(0)
                self.logger.experiment.add_images(
                    "train/final_warp_view", final_warp, self.global_step
                )

                if self.opts.use_consistentLoss:
                    # Visualizing reprojected depth image of each input view
                    list_inputDepth_rpj = []
                    for i in range(self.opts.n_views - 1):
                        input_depth_rpj = visualize_depth(results["input_depths_0"][i][0])
                        list_inputDepth_rpj.append(input_depth_rpj)
                    stack_inputDepth_rpj = torch.stack(
                        [list_inputDepth_rpj[i] for i in range(len(list_inputDepth_rpj))]
                    )  # (
                    self.logger.experiment.add_images(
                        "train/inputDepth_rpj", stack_inputDepth_rpj, self.global_step
                    )

                    # Visualizing each reprojected input image using estimated novel view and depth map
                    list_inputView_rpj = []
                    for i in range(self.opts.n_views - 1):
                        inputView_rpj = self.unpreprocess(results[f"reconstructed_input_0"][i][0])
                        list_inputView_rpj.append(inputView_rpj)
                    stack_inputView_rpj = torch.stack(
                        [list_inputView_rpj[i] for i in range(len(list_inputView_rpj))]
                    )  # (
                    self.logger.experiment.add_images(
                        "train/inputView_rpj", stack_inputView_rpj, self.global_step
                    )

            depth_pred = results["depth_0"]
            depth_gt = depths["level_0"]
            mask = masks["level_0"]
            abs_err = abs_error(
                depth_pred, depth_gt, mask
            ).mean()
            self.log("train/abs_err", abs_err, on_step=True, prog_bar=True)
            self.log("train/acc_1mm", acc_threshold(depth_pred, depth_gt, mask, 1).mean(), on_epoch=True)
            self.log("train/acc_2mm", acc_threshold(depth_pred, depth_gt, mask, 2).mean(), on_epoch=True)
            self.log("train/acc_4mm", acc_threshold(depth_pred, depth_gt, mask, 4).mean(), on_epoch=True)

            novel_view = self.unpreprocess(results["warp_view_0"][0])
            gt_novel_view = self.unpreprocess(imgs[0, 0])
            self.log("train/ssim", ssim(novel_view, gt_novel_view), on_epoch=True)
            # log["train/lpis"] = lpips(novel_view, gt_novel_view)
            self.log("train/psnr", psnr(novel_view, gt_novel_view), on_epoch=True)

        return loss

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=self.opts.batch_size,
            pin_memory=True,
        )

    def validation_step(self, batch, batch_nb):
        (
            imgs,
            proj_mats,
            depths,
            masks,
            init_depth_min,
            depth_interval,
            proj_mats_ref2inputs
        ) = self.decode_batch(batch)
        results = self(imgs, proj_mats, proj_mats_ref2inputs, init_depth_min, depth_interval)

        loss = self.getLoss(opts.loss_type, results, imgs, depths, masks)
        sync_log = True if self.opts.num_gpus > 1 else False
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=sync_log)

        if batch_nb == 0:
            img_ = self.unpreprocess(imgs[0, 0]).cpu()  # batch 0, ref image
            depth_gt_ = visualize_depth(depths["level_0"][0])
            depth_pred_ = visualize_depth(results["depth_0"][0] * masks["level_0"][0])
            prob = visualize_prob(results["confidence_0"][0] * masks["level_0"][0])
            stack = torch.stack([img_, depth_gt_, depth_pred_, prob])  # (4, 3, H, W)
            self.logger.experiment.add_images(
                "val/image_GT_pred_prob", stack, self.global_step
            )

            list_img = []
            for i in range(self.opts.n_views):
                img_i = self.unpreprocess(imgs[0, i]).cpu()
                list_img.append(img_i)
            stack_imgs = torch.stack([list_img[i] for i in range(self.opts.n_views)])
            self.logger.experiment.add_images(
                "val/GT_inputviews", stack_imgs, self.global_step
            )

            final_warp = self.unpreprocess(results["warp_view_0"][0]).unsqueeze(0)
            self.logger.experiment.add_images(
                "val/final_warp_view", final_warp, self.global_step
            )

        depth_pred = results["depth_0"]
        depth_gt = depths["level_0"]
        mask = masks["level_0"]

        novel_view = self.unpreprocess(results["warp_view_0"][0])
        gt_novel_view = self.unpreprocess(imgs[0, 0])
        self.log('val/abs_err', abs_error(depth_pred, depth_gt, mask).mean(), sync_dist=sync_log)
        self.log('val/acc_1mm', acc_threshold(depth_pred, depth_gt, mask, 1).mean(), sync_dist=sync_log)
        self.log('val/acc_2mm', acc_threshold(depth_pred, depth_gt, mask, 2).mean(), sync_dist=sync_log)
        self.log('val/acc_4mm', acc_threshold(depth_pred, depth_gt, mask, 4).mean(), sync_dist=sync_log)
        self.log('val/ssim', ssim(novel_view, gt_novel_view))
        self.log('val/psnr', psnr(novel_view, gt_novel_view))

        return loss

if __name__ == "__main__":
    opts = get_opts()
    system = NovelDepthSystem(opts)

    checkpoint_callback = ModelCheckpoint(

        dirpath=os.path.join(f'{opts.ckpt_dir}/{opts.exp_name}'),
        filename = "{epoch:02d}",
        monitor='val/loss',
        mode='max',
        save_top_k=2
    )
    bar = TQDMProgressBar(refresh_rate=100 if opts.num_gpus > 1 else 1)
    logger = TensorBoardLogger(save_dir=f'{opts.log_dir}', name=opts.exp_name)

    trainer = Trainer(
        max_epochs=opts.num_epochs,
	    callbacks=[checkpoint_callback,bar],
        logger=logger,
        enable_model_summary=True,
        gpus=opts.num_gpus,
        strategy=DDPPlugin(find_unused_parameters=True), 
        num_sanity_val_steps=0 if opts.num_gpus > 1 else 5,
        gradient_clip_val=0.5,
        benchmark=True,
    )
    # Add a comment ^^
    trainer.fit(system)
    trainer.save_checkpoint(os.path.join(f'{opts.ckpt_dir}/{opts.exp_name}', 'epoch_final.ckpt'))
