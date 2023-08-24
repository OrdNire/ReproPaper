import torch
import argparse
import os
import time
import tqdm

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import config
import data_utils
import evaluate
from model import AMF

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default= 0.00025, help="learning rate")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size for training")
parser.add_argument("--epochs", type=int, default=1500, help="training epochs")
parser.add_argument("--top_k", type=int, default=100, help="compute metrics@top_k")
parser.add_argument("--epoch_adv", type=int, default=1000, help="adversarial training epoch")
parser.add_argument("--lambda_param", type=float, default=0.0, help="lambda parameters")
parser.add_argument("--lambda_adv", type=float, default=1, help="adversarial loss ratio")
parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon")
parser.add_argument("--embed_size", type=float, default=64, help="embedding size")

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
train_user_items, test_user_items, train_pair, test_pair, user_num, item_num = data_utils.load_all_data()

train_dataset = data_utils.APRDataset(user_num, item_num, train_user_items, train_pair)
test_dataset = data_utils.APRDataset(user_num, item_num, test_user_items, test_pair, is_training=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

######################### LOAD MODELS #########################
if os.path.exists(config.AMF_MODEL):
    model = torch.load(config.AMF_MODEL)
else:
    model = AMF(user_num, item_num, args.embed_size, args.lambda_param, args.lambda_adv, args.epsilon, args.epoch_adv).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

######################### TRAINING #########################
for epoch in range(args.epochs):
    avg_loss = 0.0
    model.train()
    start_time = time.time()

    t = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)

    for user, item_i, item_j in t:
        user = user.to(device)
        item_i = item_i.to(device)
        item_j = item_j.to(device)

        model.zero_grad()
        loss = model(user, item_i, item_j, epoch)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    HR, NDCG = evaluate.metrics(test_loader, model.embed_user.detach(), model.embed_item.detach(),
                     train_user_items, device, args.top_k)
    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(HR, NDCG))
    print("loss: {}".format(avg_loss / len(train_loader)))