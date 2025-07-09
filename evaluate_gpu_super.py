import cv2
import scipy.io
import torch
import numpy as np
import time
import os
from torchvision import datasets
import functools
from models.matching import Matching
from models.utils import read_image

start_time = time.time()
# 无人机到卫星
gallery_name = 'gallery_satellite'
query_name = 'query_drone_new'
data_dir = '/home/lenovo/桌面/data/test'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [gallery_name, query_name]}
query_paths = [path for path, _ in image_datasets[query_name].imgs]
gallery_paths = [path for path, _ in image_datasets[gallery_name].imgs]

#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl):
    if qf.device != gf.device:
        gf = gf.to(qf.device)
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
feature_cache = {}

@functools.lru_cache(maxsize=None)
def preprocess_image_cached(image_path):
    img, inp, _ = read_image(image_path, device, resize, 0, False)
    return inp

def superglue_match(query_path, gallery_path):
    inp1 = preprocess_image_cached(query_path)
    inp2 = preprocess_image_cached(gallery_path)
    with torch.no_grad():
        pred = matching({'image0': inp1, 'image1': inp2})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts1, kpts2 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

    valid = conf > 0.8
    # valid = matches > -1
    mkpts1 = kpts1[valid]
    return len(mkpts1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def rerank(qf, ql, gf, gl):
    qf = qf.to(device)
    gf = gf.to(device)
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0
    for i in range(len(ql)):
        mquery_index1 = np.argwhere(ql == ql[i])  # 返回非0元素的索引
        mq = torch.mean(qf[mquery_index1, :], dim=0)
        ap_tmp, CMC_tmp = evaluate(mq, ql[i], gf, gl)
        if CMC_tmp[0] == -1:
            continue
        if CMC_tmp[0] == 0:
            scores = torch.mm(gf, mq.view(-1, 1)).squeeze(1).cpu().numpy()
            top5_indices = np.argsort(-scores)[:5]
            loftr_scores = []
            query_path = query_paths[i]
            for idx in top5_indices:
                gallery_path = gallery_paths[idx]
                good = superglue_match(query_path, gallery_path)
                loftr_scores.append(good)
            sorted_top5_indices = top5_indices[np.argsort(-np.array(loftr_scores))]
            index = np.concatenate([sorted_top5_indices, np.delete(np.arange(len(gl)), top5_indices)])
            ap_tmp, CMC_tmp = compute_mAP(index, np.argwhere(gl == ql[i]), np.argwhere(gl == -1))
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(ql)  # average CMC
    return CMC, ap/len(ql)
######################################################################
# result = scipy.io.loadmat('pytorch_result.mat')
# query_feature = torch.FloatTensor(result['query_f'])
# query_label = result['query_label'][0]
# gallery_feature = torch.FloatTensor(result['gallery_f'])
# gallery_label = result['gallery_label'][0]
multi = os.path.isfile('multi_query.mat')

# if multi:
m_result = scipy.io.loadmat('multi_query.mat')
mquery_feature = torch.FloatTensor(m_result['mquery_f'])
mquery_label = m_result['mquery_label'][0]
mquery_feature = mquery_feature.cuda()
gallery_label = m_result['gallery_label'][0]
gallery_feature = torch.FloatTensor(m_result['gallery_f'])

query_feature = mquery_feature.cuda()
# query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
print(gallery_feature.shape)
# print(gallery_feature[0, :])
# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# # print(query_label)
# for i in range(len(query_label)):
#     ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
#     if CMC_tmp[0] == -1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     # print(i, CMC_tmp[0])
#
# CMC = CMC.float()
# CMC = CMC/len(query_label)  # average CMC
# # print(round(len(gallery_label)*0.01))
# print('Recall@1:%.2f Recall@5:%.2f  Recall@top1:%.2f AP:%.2f' % (CMC[0]*100, CMC[4]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100))Recall@10:%.2f   , CMC[9]*100

resize = [640, 480]
superglue_weights = 'outdoor'
max_kps = 1024
kps_threshold = 0.005
nms_radius = 4
sinkhorn_iterations = 20
match_threshold = 0.2

config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': kps_threshold,
            'max_keypoints': max_kps
        },
        'superglue': {
            'weights': superglue_weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
}
matching = Matching(config).eval().to(device)

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
# if multi:
CMC1, ap1 = rerank(mquery_feature, mquery_label, gallery_feature, gallery_label)
print('multi Rank@1:%.2f Rank@5:%.2f  mAP:%.2f' % (CMC1[0] * 100, CMC1[4] * 100, ap1 * 100))
   # for i in range(len(mquery_label)):
   #     mquery_index1 = np.argwhere(mquery_label == mquery_label[i])  # 返回非0元素的索引
   #
   #     mq = torch.mean(mquery_feature[mquery_index1, :], dim=0)
   #     index = evaluate(mq, mquery_label[i], gallery_feature, gallery_label)
   #     top_k_indices = np.sort(index[:top_k])
   #     top_k_gallery_paths = [gallery_paths[idx] for idx in top_k_indices]
   #     d2_net_features = extract_features(d2_net_model, top_k_gallery_paths)
   #     d2_net_features = d2_net_features.to(torch.device("cuda"))
   #     d2_net_features = d2_net_features.view(5, -1)
   #     # print(d2_net_features.shape)
   #     query = mq.view(-1, 1)
   #     query = query.to(torch.device("cuda"))
   #     query = query.squeeze(1)
   #     # print(query.shape)
   #     score = torch.mm(d2_net_features, query.unsqueeze(1))
   #     score = score.squeeze(1).cpu()
   #     score = score.numpy()
   #
   #     refined_index = np.argsort(score)[::-1]
   #     query_index = np.argwhere(gallery_label == mquery_label[i])
   #     good_index = query_index
   #     junk_index = np.argwhere(gallery_label == -1)
   #     ap_tmp, CMC_tmp = compute_mAP(refined_index, good_index, junk_index)
   #     if CMC_tmp[0] == -1:
   #         continue
   #     CMC = CMC + CMC_tmp
   #     ap += ap_tmp
   #
   # CMC = CMC.float()
   # CMC = CMC/len(mquery_label)  # average CMC
   # print('multi Rank@1:%.2f Rank@5:%.2f mAP:%.2f' % (CMC[0]*100, CMC[4]*100, ap/len(mquery_label)*100))# Rank@10:%.2f  CMC[9]*100,

end_time = time.time()
total_time = end_time - start_time
print(f"程序运行时间：{total_time:.2f}秒")

# multiple-query evaluation is not used.
# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# if multi:
#    for i in range(len(query_label)):
#        mquery_index1 = np.argwhere(mquery_label == query_label[i])  # 返回非0元素的索引
#        mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
#        mquery_index = np.intersect1d(mquery_index, mquery_index2)   # 求数组交集
#        mq = torch.mean(mquery_feature[mquery_index1, :], dim=0)
#        ap_tmp, CMC_tmp = evaluate(mq, query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
#        if CMC_tmp[0] == -1:
#            continue
#        CMC = CMC + CMC_tmp
#        ap += ap_tmp
#        #print(i, CMC_tmp[0])
#    CMC = CMC.float()
#    CMC = CMC/len(query_label) #average CMC
#    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap/len(query_label)))

