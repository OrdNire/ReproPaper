import torch
import numpy as np

def hr(gt_item, recommends):
    if gt_item in recommends:
        return 1
    return 0

def ndcg(gt_item, recommends):
    if gt_item in recommends:
        return np.reciprocal(np.log2(recommends.index(gt_item) + 2))
    return 0

def metrics(model, test_loader, device,top_k):
    HR, NDCG = [], []
    with torch.no_grad():
        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)

            prediction = model(user, item)

            _, indices = torch.topk(prediction, top_k)
            # 转换为topk推荐列表
            recommends = torch.take(item, indices).detach().cpu().numpy().tolist()

            # ground truth
            gt_item = item[0].item()
            HR.append(hr(gt_item, recommends))
            NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)