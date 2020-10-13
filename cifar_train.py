import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Variable
from utils.average_meter import AverageMeter
from utils.data_prefetcher import DataPrefetcher
from utils.logutil import get_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from models.channel_distillation import *
import losses
from cifar_config import Config
from utils.metric import accuracy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    setup_seed(2020)

    logger = get_logger(__name__, Config.log)

    Config.gpus = torch.cuda.device_count()
    logger.info("use {} gpus".format(Config.gpus))
    config = {  # 用类名直接调用__dict__，会输出由该类中所有类属性组成的字典
        key: value
        for key, value in Config.__dict__.items() if not key.startswith("__")
    }
    logger.info(f"args: {config}")

    start_time = time.time()

    # dataset and dataloader
    logger.info("start loading data")

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_dataset = CIFAR100(
        Config.train_dataset_path,
        train=True,
        transform=train_transform,
        download=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    val_dataset = CIFAR100(
        Config.val_dataset_path,
        train=False,
        transform=val_transform,
        download=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    logger.info("finish loading data")

    if Config.baseline:
        net = ChannelDistillWRN1628(Config.num_classes)  # 返回了(ss, ts) 学生网络和预训练的教师网络
        net = nn.DataParallel(net).cuda()  # ChannelDistillResNet50152( (student): ResNet() (teacher): ResNet())

        optimizer_s = torch.optim.SGD(net.module.student.parameters(), lr=Config.lr_logit, momentum=0.9,
                                      weight_decay=1e-4)
        optimizer_t = torch.optim.SGD(net.module.teacher.parameters(), lr=Config.lr_logit, momentum=0.9,
                                      weight_decay=1e-4)

        scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=[150, 225], gamma=0.1)
        scheduler_t = torch.optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=[150, 225], gamma=0.1)

        optimizer = [optimizer_s, optimizer_t]
        scheduler = [scheduler_s, scheduler_t]

        # loss and optimizer
        criterion = losses.__dict__["CELoss"]().cuda()

        start_epoch = 1
        # resume training
        if os.path.exists(Config.resume):
            pass

        if not os.path.exists(Config.checkpoints):
            os.makedirs(Config.checkpoints)

        logger.info('start training')
        best_stu_acc = 0.
        best_tea_acc = 0.
        for epoch in range(start_epoch, Config.epochs + 1):
            logger.info(f"train:\n")
            prec1_s, prec1_t, prec5_s, prec5_t, loss_s, loss_t = train_baseline(train_loader, net, criterion,
                                                                                optimizer, scheduler, epoch, logger)
            logger.info(f"Student ---> train: epoch {epoch:0>3d}, top1 acc: {prec1_s:.2f}%, top5 acc: {prec5_s:.2f}%\n")
            logger.info(f"Teacher ---> train: epoch {epoch:0>3d}, top1 acc: {prec1_t:.2f}%, top5 acc: {prec5_t:.2f}%\n")

            logger.info(f"val:\n")
            prec1_s, prec5_s, prec1_t, prec5_t = validate(val_loader, net)
            logger.info(f"Student ---> val: epoch {epoch:0>3d}, top1 acc: {prec1_s:.2f}%, top5 acc: {prec5_s:.2f}%\n")
            logger.info(f"Teacher ---> val: epoch {epoch:0>3d}, top1 acc: {prec1_t:.2f}%, top5 acc: {prec5_t:.2f}%\n")

            # remember best prec@1 and save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "acc": prec1_s,
                    "loss": loss_s,
                    "lr": scheduler[0].get_lr()[0],
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer[0].state_dict(),
                    "scheduler_state_dict": scheduler[0].state_dict(),
                },
                os.path.join(Config.checkpoints, "stu_base_latest.pth")
            )
            if prec1_s > best_stu_acc:
                shutil.copyfile(os.path.join(Config.checkpoints, "stu_base_latest.pth"),
                                os.path.join(Config.checkpoints, "stu_base_best.pth"))
                best_stu_acc = prec1_s

            torch.save(
                {
                    "epoch": epoch,
                    "acc": prec1_t,
                    "loss": loss_t,
                    "lr": scheduler[1].get_lr()[0],
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer[1].state_dict(),
                    "scheduler_state_dict": scheduler[1].state_dict(),
                },
                os.path.join(Config.checkpoints, "tea_base_latest.pth")
            )
            if prec1_t > best_tea_acc:
                shutil.copyfile(os.path.join(Config.checkpoints, "tea_base_latest.pth"),
                                os.path.join(Config.checkpoints, "tea_base_best.pth"))
                best_tea_acc = prec1_t

        training_time = (time.time() - start_time) / 3600
        logger.info(f"finish training\n")
        logger.info(
            f"Stu -> best acc: {best_stu_acc:.2f}%, Tea -> best acc: {best_tea_acc:.2f}%, total training time: {training_time:.2f} hours")
    else:
        # network
        net = ChannelDistillWRN1628(Config.num_classes)  # 返回了(ss, ts) 学生网络和预训练的教师网络
        # net = ChannelDistillResNet50152(Config.num_classes, Config.dataset_type)  # 返回了(ss, ts) 学生网络和预训练的教师网络
        net = nn.DataParallel(net).cuda()  # ChannelDistillResNet50152( (student): ResNet() (teacher): ResNet())

        discriminator = DiscriminatorStudentTeacher(128, Config.model_type).cuda()  # WRN最后剩下128
        # discriminator = DiscriminatorStudentTeacher(2048, Config.model_type).cuda()

        # loss and optimizer
        criterion = [losses.__dict__["CELoss"]().cuda(), losses.__dict__["KDLoss"](Config.T).cuda(),
                     torch.nn.MSELoss().cuda()]

        # 优化学生和老师 -> feature extracter
        optimizer_logit = [torch.optim.SGD(net.module.student.parameters(), lr=Config.lr_logit, momentum=0.9,
                                           weight_decay=1e-4),
                           torch.optim.SGD(net.module.teacher.parameters(), lr=Config.lr_logit, momentum=0.9,
                                           weight_decay=1e-4)]  # g1, g2
        scheduler_logit = [torch.optim.lr_scheduler.MultiStepLR(optimizer_logit[0], milestones=[150, 225], gamma=0.1),
                           torch.optim.lr_scheduler.MultiStepLR(optimizer_logit[1], milestones=[150, 225], gamma=0.1)]

        # 优化学生和教师及其他们的判别器-> feature extracter and D1 and D2
        optimizer_g1_fmap = torch.optim.Adam(net.module.student.parameters(), lr=Config.lr_fmap, weight_decay=1e-1)
        optimizer_d1_fmap = torch.optim.Adam(discriminator.discri_s.parameters(), lr=Config.lr_fmap, weight_decay=1e-1)
        scheduler_g1_fmap = torch.optim.lr_scheduler.MultiStepLR(optimizer_g1_fmap, milestones=[75, 150], gamma=0.1)
        scheduler_d1_fmap = torch.optim.lr_scheduler.MultiStepLR(optimizer_d1_fmap, milestones=[75, 150], gamma=0.1)

        optimizer_s_fmap = [optimizer_g1_fmap, optimizer_d1_fmap]  # g1, d1
        scheduler_s_fmap = [scheduler_g1_fmap, scheduler_d1_fmap]

        optimizer_g2_fmap = torch.optim.Adam(net.module.teacher.parameters(), lr=Config.lr_fmap, weight_decay=1e-1)
        optimizer_d2_fmap = torch.optim.Adam(discriminator.discri_t.parameters(), lr=Config.lr_fmap, weight_decay=1e-1)
        scheduler_g2_fmap = torch.optim.lr_scheduler.MultiStepLR(optimizer_g2_fmap, milestones=[75, 150], gamma=0.1)
        scheduler_d2_fmap = torch.optim.lr_scheduler.MultiStepLR(optimizer_d2_fmap, milestones=[75, 150], gamma=0.1)

        optimizer_t_fmap = [optimizer_g2_fmap, optimizer_d2_fmap]  # g2, d2
        scheduler_t_fmap = [scheduler_g2_fmap, scheduler_d2_fmap]

        # only evaluate
        if Config.evaluate:
            pass

        start_epoch = 1
        # resume training
        if os.path.exists(Config.resume):
            pass

        if not os.path.exists(Config.checkpoints):
            os.makedirs(Config.checkpoints)

        logger.info('start training')
        best_stu_acc = 0.
        best_tea_acc = 0.
        for epoch in range(start_epoch, Config.epochs + 1):
            logger.info(f"train:\n")
            prec1_s, prec1_t, prec5_s, prec5_t, loss_s, loss_t = train(train_loader, net, discriminator, criterion,
                                                                       optimizer_logit,
                                                                       scheduler_logit,
                                                                       optimizer_s_fmap, scheduler_s_fmap,
                                                                       optimizer_t_fmap,
                                                                       scheduler_t_fmap, epoch,
                                                                       logger)
            logger.info(f"Student ---> train: epoch {epoch:0>3d}, top1 acc: {prec1_s:.2f}%, top5 acc: {prec5_s:.2f}%\n")
            logger.info(f"Teacher ---> train: epoch {epoch:0>3d}, top1 acc: {prec1_t:.2f}%, top5 acc: {prec5_t:.2f}%\n")

            logger.info(f"val:\n")
            prec1_s, prec5_s, prec1_t, prec5_t = validate(val_loader, net)
            logger.info(f"Student ---> val: epoch {epoch:0>3d}, top1 acc: {prec1_s:.2f}%, top5 acc: {prec5_s:.2f}%\n")
            logger.info(f"Teacher ---> val: epoch {epoch:0>3d}, top1 acc: {prec1_t:.2f}%, top5 acc: {prec5_t:.2f}%\n")

            # remember best prec@1 and save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "acc": prec1_s,
                    "loss": loss_s,
                    "lr_logit": scheduler_logit[0].get_lr()[0],
                    "lr_g": scheduler_s_fmap[0].get_lr()[0],
                    "lr_d": scheduler_s_fmap[1].get_lr()[0],
                    "model_state_dict": net.state_dict(),
                    "optimizer_logit_state_dict": optimizer_logit[0].state_dict(),
                    "optimizer_fmap_g_state_dict": optimizer_s_fmap[0].state_dict(),
                    "optimizer_fmap_d_state_dict": optimizer_s_fmap[1].state_dict(),
                    "scheduler_logit_state_dict": scheduler_logit[0].state_dict(),
                    "scheduler_g_state_dict": scheduler_s_fmap[0].state_dict(),
                    "scheduler_d_state_dict": scheduler_s_fmap[1].state_dict(),
                },
                os.path.join(Config.checkpoints, "stu_latest.pth")
            )
            if prec1_s > best_stu_acc:
                shutil.copyfile(os.path.join(Config.checkpoints, "stu_latest.pth"),
                                os.path.join(Config.checkpoints, "stu_best.pth"))
                best_stu_acc = prec1_s

            torch.save(
                {
                    "epoch": epoch,
                    "acc": prec1_t,
                    "loss": loss_t,
                    "lr_logit": scheduler_logit[1].get_lr()[0],
                    "lr_g": scheduler_t_fmap[0].get_lr()[0],
                    "lr_d": scheduler_t_fmap[1].get_lr()[0],
                    "model_state_dict": net.state_dict(),
                    "optimizer_logit_state_dict": optimizer_logit[1].state_dict(),
                    "optimizer_fmap_g_state_dict": optimizer_t_fmap[0].state_dict(),
                    "optimizer_fmap_d_state_dict": optimizer_t_fmap[1].state_dict(),
                    "scheduler_logit_state_dict": scheduler_logit[1].state_dict(),
                    "scheduler_g_state_dict": scheduler_t_fmap[0].state_dict(),
                    "scheduler_d_state_dict": scheduler_t_fmap[1].state_dict(),
                },
                os.path.join(Config.checkpoints, "tea_latest.pth")
            )
            if prec1_t > best_tea_acc:
                shutil.copyfile(os.path.join(Config.checkpoints, "tea_latest.pth"),
                                os.path.join(Config.checkpoints, "tea_best.pth"))
                best_tea_acc = prec1_t

        training_time = (time.time() - start_time) / 3600
        logger.info(f"finish training\n")
        logger.info(
            f"Stu -> best acc: {best_stu_acc:.2f}%, Tea -> best acc: {best_tea_acc:.2f}%, total training time: {training_time:.2f} hours")


def train(train_loader, net, discriminator, criterion, optimizer_logit, scheduler_logit,
          optimizer_s_fmap, scheduler_s_fmap, optimizer_t_fmap, scheduler_t_fmap, epoch,
          logger):
    top1 = [AverageMeter(), AverageMeter()]
    top5 = [AverageMeter(), AverageMeter()]
    loss_total = [AverageMeter(), AverageMeter()]

    # switch to train mode
    net.train()

    iters = len(train_loader.dataset) // Config.batch_size
    prefetcher = DataPrefetcher(train_loader)  # 加速数据读取
    inputs, labels = prefetcher.next()

    iter = 1
    while inputs is not None:
        inputs, labels = inputs.float().cuda(), labels.cuda()

        # Adversarial ground truths
        valid = Variable(torch.cuda.FloatTensor(inputs.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(inputs.shape[0], 1, 1, 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        # zero the parameter gradients
        # student
        optimizer_logit[0].zero_grad()  # s_g_logit
        optimizer_s_fmap[0].zero_grad()  # s_g_fmap
        # teacher
        optimizer_logit[1].zero_grad()  # t_g_logit
        optimizer_t_fmap[0].zero_grad()  # t_g_fmap

        # forward + backword + optimize
        stu_outputs, tea_outputs = net(inputs)  # [x1, x2, x3, x4, x]

        # student
        loss_stu_logit = criterion[0](stu_outputs[-1], labels) + criterion[1](stu_outputs[-1], tea_outputs[-1].detach())

        loss_stu_g = criterion[2](valid, discriminator.discri_s(stu_outputs[-2]))
        loss_stu = loss_stu_logit + loss_stu_g

        # teacher
        loss_tea_logit = criterion[0](tea_outputs[-1], labels) + criterion[1](tea_outputs[-1], stu_outputs[-1].detach())
        loss_tea_g = criterion[2](valid, discriminator.discri_t(tea_outputs[-2]))
        loss_tea = loss_tea_logit + loss_tea_g

        # student
        loss_stu.backward()
        optimizer_logit[0].step()  # s_g_logit
        optimizer_s_fmap[0].step()  # s_g_fmap

        # teacher
        loss_tea.backward()
        optimizer_logit[1].step()  # t_g_logit
        optimizer_t_fmap[0].step()  # t_g_fmap

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_s_fmap[1].zero_grad()  # s_d
        optimizer_t_fmap[1].zero_grad()  # t_d

        # discriminator loss
        # student
        loss_stu_d = criterion[2](valid, discriminator.discri_s(tea_outputs[-2].detach())) + \
                     criterion[2](fake, discriminator.discri_s(stu_outputs[-2].detach()))
        # teacher
        loss_tea_d = criterion[2](valid, discriminator.discri_t(stu_outputs[-2].detach())) + \
                     criterion[2](fake, discriminator.discri_t(tea_outputs[-2].detach()))

        # student
        loss_stu_d.backward()
        optimizer_s_fmap[1].step()

        # teacher
        loss_tea_d.backward()
        optimizer_t_fmap[1].step()

        # student
        prec_s = accuracy(stu_outputs[-1], labels, topk=(1, 5))
        top1[0].update(prec_s[0].item(), inputs.size(0))
        top5[0].update(prec_s[1].item(), inputs.size(0))
        loss_total[0].update(loss_stu.item(), inputs.size(0))

        # teacher
        prec_t = accuracy(tea_outputs[-1], labels, topk=(1, 5))
        top1[1].update(prec_t[0].item(), inputs.size(0))
        top5[1].update(prec_t[1].item(), inputs.size(0))
        loss_total[1].update(loss_tea.item(), inputs.size(0))

        inputs, labels = prefetcher.next()  # 取下一批数据

        if iter % 20 == 0:
            loss_log = f"train: epoch {epoch:0>3d}, iter [{iter:0>4d}, {iters:0>4d}]\n"
            loss_log += f"Student detail:\n "
            loss_log += f"top1 acc: {prec_s[0]:.2f}%, top5 acc: {prec_s[1]:.2f}%, "
            loss_log += f"loss_total: {loss_stu.item():3f}, "
            loss_log += f"loss_logit: {loss_stu_logit.item():3f} "
            loss_log += f"loss_g: {loss_stu_g.item():3f} "
            loss_log += f"loss_d: {loss_stu_d.item():3f} "

            loss_log += f"\nTeacher detail:\n "
            loss_log += f"top1 acc: {prec_t[0]:.2f}%, top5 acc: {prec_t[1]:.2f}%, "
            loss_log += f"loss_total: {loss_tea.item():3f}, "
            loss_log += f"loss_logit: {loss_tea_logit.item():3f} "
            loss_log += f"loss_g: {loss_tea_g.item():3f} "
            loss_log += f"loss_d: {loss_tea_d.item():3f} "
            logger.info(loss_log)
        iter += 1

    scheduler_logit[0].step()
    scheduler_s_fmap[0].step()
    scheduler_logit[1].step()
    scheduler_t_fmap[0].step()
    scheduler_s_fmap[1].step()
    scheduler_t_fmap[1].step()

    return top1[0].avg, top1[1].avg, top5[0].avg, top5[1].avg, loss_total[0].avg, loss_total[1].avg


def train_baseline(train_loader, net, criterion, optimizer, scheduler, epoch, logger):
    top1 = [AverageMeter(), AverageMeter()]
    top5 = [AverageMeter(), AverageMeter()]
    loss_total = [AverageMeter(), AverageMeter()]

    # switch to train mode
    net.train()

    iters = len(train_loader.dataset) // Config.batch_size
    prefetcher = DataPrefetcher(train_loader)  # 加速数据读取
    inputs, labels = prefetcher.next()

    iter = 1
    while inputs is not None:
        inputs, labels = inputs.float().cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        # forward + backword + optimize
        stu_outputs, tea_outputs = net(inputs)  # [x1, x2, x3, x4, x]

        loss_stu = criterion(stu_outputs[-1], labels)
        loss_tea = criterion(tea_outputs[-1], labels)

        loss_stu.backward()
        loss_tea.backward()
        optimizer[0].step()
        optimizer[1].step()

        # student
        prec_s = accuracy(stu_outputs[-1], labels, topk=(1, 5))
        top1[0].update(prec_s[0].item(), inputs.size(0))
        top5[0].update(prec_s[1].item(), inputs.size(0))
        loss_total[0].update(loss_stu.item(), inputs.size(0))

        # teacher
        prec_t = accuracy(tea_outputs[-1], labels, topk=(1, 5))
        top1[1].update(prec_t[0].item(), inputs.size(0))
        top5[1].update(prec_t[1].item(), inputs.size(0))
        loss_total[1].update(loss_tea.item(), inputs.size(0))

        inputs, labels = prefetcher.next()  # 取下一批数据

        if iter % 20 == 0:
            loss_log = f"train: epoch {epoch:0>3d}, iter [{iter:0>4d}, {iters:0>4d}]\n"
            loss_log += f"Student detail:\n "
            loss_log += f"top1 acc: {prec_s[0]:.2f}%, top5 acc: {prec_s[1]:.2f}%, "
            loss_log += f"loss_s: {loss_stu.item():3f}, "

            loss_log += f"\nTeacher detail:\n "
            loss_log += f"top1 acc: {prec_t[0]:.2f}%, top5 acc: {prec_t[1]:.2f}%, "
            loss_log += f"loss_t: {loss_tea.item():3f}, "
            logger.info(loss_log)
        iter += 1
    scheduler[0].step()
    scheduler[1].step()
    return top1[0].avg, top1[1].avg, top5[0].avg, top5[1].avg, loss_total[0].avg, loss_total[1].avg


def validate(val_loader, net):
    top1 = [AverageMeter(), AverageMeter()]
    top5 = [AverageMeter(), AverageMeter()]

    # switch to evaluate mode
    net.eval()

    prefetcher = DataPrefetcher(val_loader)
    inputs, labels = prefetcher.next()
    with torch.no_grad():
        while inputs is not None:
            inputs = inputs.float().cuda()
            labels = labels.cuda()

            stu_outputs, tea_outputs = net(inputs)

            pred_s = accuracy(stu_outputs[-1], labels, topk=(1, 5))
            pred_t = accuracy(tea_outputs[-1], labels, topk=(1, 5))

            top1[0].update(pred_s[0].item(), inputs.size(0))
            top5[0].update(pred_s[1].item(), inputs.size(0))

            top1[1].update(pred_t[0].item(), inputs.size(0))
            top5[1].update(pred_t[1].item(), inputs.size(0))

            inputs, labels = prefetcher.next()

    return top1[0].avg, top5[0].avg, top1[1].avg, top5[1].avg


if __name__ == '__main__':
    main()  # CUDA_VISIBLE_DEVICES=0 python3 ./cifar_train.py

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--log', default="./log", help='Path to save log')
    # parser.add_argument('--cpt', default="./checkpoints", help='Path to store model')
    # parser.add_argument('--resume', default="./checkpoints/latest.pth", help='resume from checkpoint')
    # parser.add_argument('--dataset', default='cifar', type=str, help='dateset type')
    # parser.add_argument('--train_root', default='./data/CIFAR100', type=str, help='train_dataset_path')
    # parser.add_argument('--val_root', default='./data/CIFAR100', type=str, help='val_dataset_path')
    # parser.add_argument('--eval', '-e', action='store_true', help='only evaluate')
    # parser.add_argument('--arch', '-a', default='wrn', type=str, help='model type')
    # parser.add_argument('--temp', default=3.0, type=float, help='temperature scaling')
    # parser.add_argument('--num_works', default=4, type=int)
    # parser.add_argument('--epochs', default=300, type=int, help='total epochs to run')
    # parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    # parser.add_argument('--lr_logit', default=0.1, type=float)
    # parser.add_argument('--lr_fmap', default=0.00002, type=float)
    # parser.add_argument('--seed', default=None, type=int)
    # # parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    # # parser.add_argument('--gpu', default='0', type=str)
    # # parser.add_argument('--dataset', default='cifar10', type=str)
    # # parser.add_argument('--mode', type=str)
    # # parser.add_argument('--lr_step', type=int, help='step size for StepLR')
    # # parser.add_argument('--lr_gamma', type=float, help='gamma for StepLR')
    # # parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
    #
    # args = parser.parse_args()
    # setup_seed(args.seed)
    # args.num_classes = 100
