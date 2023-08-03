import torch
import torch.nn.functional as F

def metrics(model, test_loader, device):
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for features, label in test_loader:
            features = features.to(device)
            label = label.to(device)
            predict = model(features)
            predict = (F.sigmoid(predict) > 0.5)
            num_correct += (predict == label).sum()
            num_sample += predict.size(0)

    acc = float(num_correct / num_sample)
    return acc