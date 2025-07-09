import scipy.io
import torch
import numpy as np
import time
import os
import cv2 as cv
from torchvision import datasets
start_time = time.time()
# 无人机到卫星
gallery_name = 'gallery_satellite'
query_name = 'query_drone'
# 卫星到无人机
# gallery_name = 'gallery_drone'
# query_name = 'query_satellite'

data_dir = './data1/test'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [gallery_name, query_name]}
def sift_F(image_paths):
    sift = cv.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    features = []
    for path in image_paths:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        _, des = sift.detectAndCompute(img, None)
        features.append(des)
    return features

query_paths = [path for path, _ in image_datasets[query_name].imgs]
gallery_paths = [path for path, _ in image_datasets[gallery_name].imgs]
query_sift_features = sift_F(query_paths)
gallery_sift_features = sift_F(gallery_paths)

def sift_match(query_sift, gallery_sift):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(query_sift, gallery_sift, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return len(good)

#######################################################################
# Evaluate
def evaluate(qf, ql, gf, gl):
    if qf.device != gf.device:
        gf = gf.to(qf.device)
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

    # CMC_tmp = compute_mAP(a_index, good_index, junk_index)
    # return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
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
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rerank(qf, ql, gf, gl, q_s, g_s):
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
            top10_indices = np.argsort(-torch.mm(gf, qf[i].view(-1, 1)).squeeze().cpu().numpy())[:5]
            sift_scores = []
            for idx in top10_indices:
                if q_s[i] is not None and g_s[idx] is not None:
                    sift_score = sift_match(q_s[i], g_s[idx])
                    sift_scores.append(sift_score)
                else:
                    sift_scores.append(0)

            top10_indices = top10_indices[np.argsort(-np.array(sift_scores))]
            new_index = np.concatenate([top10_indices, np.delete(np.arange(len(gl)), top10_indices)])
            ap_tmp, CMC_tmp = compute_mAP(new_index, np.argwhere(gl == ql[i]), np.argwhere(gl == -1))
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC / len(ql)  # average CMC
    return CMC, ap/len(ql)

# def rerank(qf, ql, gf, gl, q_s, g_s):
#     CMC = torch.IntTensor(len(gl)).zero_()
#     ap = 0.0
#     for i in range(len(ql)):
#         mquery_index1 = np.argwhere(ql == ql[i])  # 返回非0元素的索引
#         mq = torch.mean(qf[mquery_index1, :], dim=0)
#         ap_tmp, CMC_tmp = evaluate(mq, ql[i], gf, gl)
#         if CMC_tmp[0] == -1:
#             continue
#         if CMC_tmp[0] == 0:
#             top20_indices = np.argsort(-torch.mm(gf, qf[i].view(-1, 1)).squeeze().cpu().numpy())[:5]
#             sift_scores = []
#             for idx in top20_indices:
#                 if q_s[i] is not None and g_s[idx] is not None:
#                     sift_score = sift_match(q_s[i], g_s[idx])
#                     sift_scores.append(sift_score)
#                 else:
#                     sift_scores.append(0)
#
#             original_scores = torch.mm(gf[top20_indices], qf[i].view(-1, 1)).squeeze().cpu().numpy()
#             combined_scores = 0.5 * original_scores + 0.5 * np.array(sift_scores)
#             top20_indices = top20_indices[np.argsort(np.argsort(-combined_scores))]
#
#             new_index = np.concatenate([top20_indices, np.delete(np.arange(len(gl)), top20_indices)])
#             ap_tmp, CMC_tmp = compute_mAP(new_index, np.argwhere(gl == ql[i]), np.argwhere(gl == -1))
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#     CMC = CMC.float()
#     CMC = CMC / len(ql)  # average CMC
#     return CMC, ap/len(ql)
######################################################################
# result = scipy.io.loadmat('pytorch_result.mat')
# query_feature = torch.FloatTensor(result['query_f'])
# query_label = result['query_label'][0]
# gallery_feature = torch.FloatTensor(result['gallery_f'])
# gallery_label = result['gallery_label'][0]
multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_label = m_result['mquery_label'][0]
    # mquery_feature = mquery_feature.cuda()
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
# print(round(len(gallery_label)*0.01))
# print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (CMC[0]*100, CMC[4]*100, CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100))

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    # for i in range(len(mquery_label)):
    #     mquery_index1 = np.argwhere(mquery_label == mquery_label[i])  # 返回非0元素的索引
    #     # mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
    #     # mquery_index = np.intersect1d(mquery_index, mquery_index2)   # 求数组交集
    #     mq = torch.mean(mquery_feature[mquery_index1, :], dim=0)
    #
    #     query_path = query_paths[i]
    #     ap_tmp, CMC_tmp = evaluate(mq, mquery_label[i], gallery_feature, gallery_label)
    #     # good_index = np.argwhere(gallery_label == mquery_label[i]).flatten()
    #     # junk_index = np.argwhere(gallery_label == -1).flatten()
    #     # mask = np.in1d(index, junk_index, invert=True)
    #     # index = index[mask]
    #     # ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
    #     if CMC_tmp[0] == -1:
    #         continue
    #     CMC = CMC + CMC_tmp
    #     ap += ap_tmp
    #     # print(i, CMC_tmp[0])
    # CMC = CMC.float()
    # CMC = CMC / len(mquery_label)  # average CMC
    # print('multi Rank@1:%.2f Rank@5:%.2f Rank@10:%.2f mAP:%.2f' % (CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, ap / len(mquery_label) * 100))
    CMC1, ap1 = rerank(mquery_feature, mquery_label, gallery_feature, gallery_label, query_sift_features, gallery_sift_features)
    print('multi Rank@1:%.2f Rank@5:%.2f  mAP:%.2f' % (CMC1[0] * 100, CMC1[4] * 100, ap1 * 100))  #Rank@10:%.2f CMC1[9] * 100,

end_time = time.time()
total_time = end_time - start_time
print(f"程序运行时间：{total_time:.2f}秒")
# multiple-query evaluation is not used.
# CMC = torch.IntTensor(len(gallery_label)).zero_() 459-
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

