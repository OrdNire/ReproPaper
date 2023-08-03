import torch
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.utils.data import DataLoader, sampler
import time
import tqdm
import os

import config
import datasets
import model
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout ratio")
parser.add_argument("--emb_size", type=int, default=10, help="embedding size")
parser.add_argument("--num_neurons", type=int, default=400, help="number of Neurons per Layer")
parser.add_argument("--layer_num", type=int, default=3, help="number of MLP layers")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--use_gpu", default=True, help="use gpu or not")
args = parser.parse_args()

if args.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    cudnn.benchmark = True
    cudnn.enabled = True
else:
    device = torch.device("cpu")

######################### LOAD DATASETS #########################
Num_train = 800
features_size = pd.read_csv(config.HANDLE_FEATURES_SIZE_PATH, sep=',', header=None)
features_size = list(features_size.values.squeeze(0))
train_dataset = datasets.DeepFMDataset(config.HANDLE_TRAINING_PATH, is_training=True)
test_dataset = datasets.DeepFMDataset(config.HANDLE_TRAINING_PATH, is_training=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler.SubsetRandomSampler(range(Num_train)))
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler.SubsetRandomSampler(range(Num_train, 899)), num_workers=0)

######################### LOAD MODELS #########################
model = model.DeepFM(features_size, config.CONTINUOUS_FEATURES, config.CATEGORIAL_FEATURES,
                     args.emb_size, args.num_neurons, args.dropout, args.layer_num).to(device)

loos_fn = nn.BCEWithLogitsLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

######################### TRAINING #########################
best_acc = 0.0
for epoch in range(args.epochs):
    avg_loss = 0.0
    model.train()
    start_time = time.time()

    t = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
    for features, label in t:
        features = features.to(device)
        label = label.float().to(device)

        model.zero_grad()
        predict = model(features)
        loss = loos_fn(predict, label)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    elapsed_time = time.time() - start_time
    acc = evaluate.metrics(model, test_loader, device)
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("ACC: {:.3f}".format(acc))
    print("loss: {}".format(avg_loss / len(train_loader)))

    if acc > best_acc:
        best_acc, best_epoch = acc, epoch
        if args.out:
            if not os.path.exists(config.MODEL_PATH):
                os.mkdir(config.MODEL_PATH)
            torch.save(model,
                       '{}deepFM.pth'.format(config.MODEL_PATH))
print("End. Best epoch {:03d}: ACC = {:.3f}".format(best_epoch, best_acc))