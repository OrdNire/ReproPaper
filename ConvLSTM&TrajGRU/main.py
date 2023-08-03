import torch
from MovingMNIST import MovingMNIST
from torch.utils.data import DataLoader
import os
from Models import EF
from model_params import convlstm_encoder_params, convlstm_forecaster_params, trajgru_encoder_params, trajgru_forecaster_params
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 500
TIMESTAMP = "2023-5-26"
save_dir = "./save_model/" + TIMESTAMP
run_dir = "./runs/" + TIMESTAMP
data_dir = "./datasets/mnist"

train_set = MovingMNIST(data_dir, train=True, Norm=True)
test_set = MovingMNIST(data_dir, train=False, Norm=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


def train():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    net = EF(trajgru_encoder_params, trajgru_forecaster_params)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.to(device)

    tb = SummaryWriter(run_dir)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    if os.path.exists(os.path.join(save_dir, "checkpoint.pth.tar")):
        # 加载模型
        print("="*10 + "load model" + "="*10)
        model_info = torch.load(os.path.join(save_dir, "checkpoint.pth.tar"))
        net.load_state_dict(model_info["state_dict"])
        optimizer.load_state_dict(model_info["optimizer"])
        cur_epoch = model_info["epoch"] + 1
    else:
        cur_epoch = 0

    loss_fun = nn.MSELoss()
    loss_fun = loss_fun.to(device)

    pla_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  factor=0.5,
                                                                  patience=4,
                                                                  verbose=True)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(cur_epoch, EPOCHS + 1):
        net.train()
        t = tqdm(train_loader, leave=False, total=len(train_loader))
        for i, (inputs, targets) in enumerate(t):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fun(outputs, targets)
            loss_avg = loss.item() / BATCH_SIZE
            train_losses.append(loss_avg)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                "trainloss" : "{:6f}".format(loss_avg),
                "epoch" : "{:02d}".format(epoch)
            })

        tb.add_scalar("Trainloss", loss_avg, epoch)

        # valid
        with torch.no_grad():
            net.eval()
            t = tqdm(test_loader, leave=False, total=len(test_loader))
            for i, (inputs, targets) in enumerate(t):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                loss = loss_fun(outputs, targets)
                loss_avg = loss.item() / BATCH_SIZE
                valid_losses.append(loss_avg)
                t.set_postfix({
                    "Validloss": "{:6f}".format(loss_avg),
                    "epoch" : "{:02d}".format(epoch)
                })

            tb.add_scalar("Validloss", loss_avg, epoch)

        torch.cuda.empty_cache()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print("cur_epoch: {} total_epoch: {} train loss: {} valid loss: {}".format(
            epoch, EPOCHS, train_loss, valid_loss
        ))

        train_losses.clear()
        valid_losses.clear()

        pla_lr_scheduler.step(valid_loss)
        model_dict = {
            "epoch" : epoch,
            "state_dict" : net.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        torch.save(model_dict, os.path.join(save_dir, "checkpoint.pth.tar"))


if __name__ == '__main__':
    train()