#coding:utf-8
import os
import torch
import time
import datetime
import yaml
import torch.nn.functional as F
import torch.utils.data as data
from tools.metric import SegmentationMetric
from tools.logger import SetupLogger
from datasets.cityscapes_ import CityscapesDataset
from tools.ohem_ce_loss19_ import OhemCELoss
from tools.lr_scheduler import WarmupPolyLrScheduler
from tools.save_model import save_checkpoint
from models import MSPNet


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_parallel = torch.cuda.device_count() > 1

        # dataset and dataloader
        train_dataset = CityscapesDataset(root=cfg["train"]["cityscapes_root"],
                                          split='train',
                                          base_size=cfg["model"]["base_size"],
                                          crop_size=cfg["model"]["crop_size"])
        val_dataset = CityscapesDataset(root=cfg["train"]["cityscapes_root"],
                                        split='val',
                                        base_size=cfg["model"]["base_size"],
                                        crop_size=cfg["model"]["crop_size"])

        self.train_dataloader = data.DataLoader(dataset=train_dataset,
                                                batch_size=cfg["train"]["train_batch_size"],
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True,
                                                drop_last=True)
        self.val_dataloader = data.DataLoader(dataset=val_dataset,
                                              batch_size=cfg["train"]["valid_batch_size"],
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=True)

        self.iters_per_epoch = len(self.train_dataloader)

        # model初始化
        self.net = MSPNet.MSPNet().to(self.device)
        ##################################################### Initialize #############################
        ##################################################### Ready for public  #############################


    def train(self):
        epochs, max_iters = self.epochs, self.max_iters
        log_per_iters = self.cfg["train"]["log_iter"]
        val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.net.train()

        for _ in range(self.epochs): #self.epochs
            # Training process
            # ready for public
            pass

        # 计算此epoch的时间开销
        torch.cuda.synchronize(self.device)
######################################################### Ready for public  #############################

    def validation(self):
        is_best = False
        self.metric.reset()
        model = self.net
        model.eval()
        list_loss = []
        torch.cuda.empty_cache()

        ##################################################### Ready for public  #############################

        # average_loss = sum(list_loss) / len(list_loss)
        # self.current_mIoU = mIoU
        # logger.info(
        #     "Validation: Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}, FPS: {:.1f}".format(average_loss,
        #                                                                                             mIoU,
        #                                                                                             pixAcc,
        #                                                                                             fps))
        # if self.current_mIoU > self.best_mIoU:
        #     is_best = True
        #     self.best_mIoU = self.current_mIoU
        # if is_best:
        #     save_checkpoint(self.net, self.cfg, self.current_epoch, is_best, self.current_mIoU, self.data_parallel)


if __name__ == '__main__':
    # Set config file

    config_path = "configs/city_mspnet.yaml"
    with open(config_path, "r", encoding='utf-8') as yaml_file:
        cfg = yaml.safe_load(yaml_file.read())

    # Use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))


    # Set logger
    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["log_save_dir"],
                         distributed_rank=0,
                         filename='{}_{}_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    logger.info("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    logger.info("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    logger.info(cfg)

    # Start train
    trainer = Trainer(cfg)
    trainer.train()
