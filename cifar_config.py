# CUDA_VISIBLE_DEVICES=0 python3 ./cifar_train.py  启动
class Config:
    log = "./log"  # Path to save log
    checkpoints = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"
    evaluate = None
    dataset_type = "cifar"
    model_type = "wrn"
    train_dataset_path = './data/CIFAR100'
    val_dataset_path = './data/CIFAR100'
    baseline = True

    num_classes = 100
    num_workers = 4
    epochs = 300
    batch_size = 128
    T = 3
    lr_logit = 0.1
    lr_fmap = 0.00002
