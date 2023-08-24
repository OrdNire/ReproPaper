import torch
import tqdm
import numpy as np

def hr(gt_item, recommends):
    if gt_item in recommends:
        return 1
    return 0

def ndcg(gt_item, recommends):
    if gt_item in recommends:
        return np.reciprocal(np.log2(recommends.index(gt_item) + 2))
    return 0

def metrics(test_loader, embed_user, embed_item, train_user_items, device, top_k):
    HR, NDCG = [], []
    with torch.no_grad():
        t = tqdm.tqdm(test_loader, total=len(test_loader), leave=False)
        for user, item in t:
            # user = user.to(device)
            # item = item.to(device)
            user_emb = embed_user[user, :] # (emb_size)
            result = torch.mm(user_emb, embed_item.t()).squeeze(0) # (item_num)
            mask = torch.ones_like(result)
            index = train_user_items[user]
            for i in index:
                mask[i] = 0
            result = torch.mul(result, mask)
            _, recommends = torch.topk(result, top_k)
            recommends = recommends.detach().cpu().numpy().tolist()
            HR.append(hr(item[0], recommends))
            NDCG.append(ndcg(item[0], recommends))

    return np.mean(HR), np.mean(NDCG)

