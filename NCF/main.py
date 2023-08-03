import torch
import numpy as np
import argparse
import os
import time
import tqdm

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn

import config
import data_utils
import model
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epochs")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout ratio")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factor number")
parser.add_argument("--layer_num", type=int, default=3, help="number of MLP layers")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample negative items for testing")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--use_gpu", default=True, help="use gpu or not")
args = parser.parse_args()

if args.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    cudnn.benchmark = True
    cudnn.enabled = True
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

######################### LOAD DATASETS #########################
train_data, test_data, user_num, item_num, train_matrix = data_utils.load_all_data()

train_dataset = data_utils.NCFDataset(train_data, item_num, train_matrix, args.num_ng, is_training=True)
test_dataset = data_utils.NCFDataset(test_data, item_num, train_matrix, 0, is_training=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

######################### LOAD MODELS #########################
if config.MODEL == "NeuMF-pre":
    assert os.path.exists(config.GMF_MODEL), "lack of GMF model"
    assert os.path.exists(config.MLP_MODEL), "lack of MLP model"
    GMF_model = torch.load(config.GMF_MODEL)
    MLP_model = torch.load(config.MLP_MODEL)
else:
    GMF_model = None
    MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.layer_num, args.dropout, config.MODEL, GMF_model, MLP_model).to(device)

loss_func = nn.BCEWithLogitsLoss().to(device)

if config.MODEL == "NeuMF-pre":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


######################### TRAINING #########################
best_hr = 0
for epoch in range(args.epochs):
    avg_loss = 0.0
    model.train()
    start_time = time.time()
    train_loader.dataset.ng_sample()

    t = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)

    for user, item, label in t:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        model.zero_grad()
        prediction = model(user, item)
        loss = loss_func(prediction, label)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    HR, NDCG = evaluate.metrics(model, test_loader, device, args.top_k)
    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(HR, NDCG))
    print("loss: {}".format(avg_loss / len(train_loader)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.MODEL_PATH):
                os.mkdir(config.MODEL_PATH)
            torch.save(model,
                       '{}{}.pth'.format(config.MODEL_PATH, config.MODEL))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))