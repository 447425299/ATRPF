import scipy.io
import torch
import numpy as np
import time
import os
import cv2
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
query_paths = [path for path, _ in image_datasets[query_name].imgs]
gallery_paths = [path for path, _ in image_datasets[gallery_name].imgs]
def orb_features(img_paths):
    orb = cv2.ORB_create()
    orb_features = {}
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            orb_features[img_path] = des
        else:
            orb_features[img_path] = None
    return orb_features

def orb_match(query_des, gallery_des):
    if query_des is None or gallery_des is None:
        return 0

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(query_des, gallery_des, k=2)

    good = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return len(good)

gallery_orb_features = orb_features(gallery_paths)

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
CMC1 = torch.IntTensor(len(gallery_label)).zero_()
ap1 = 0.0
if multi:
    for i in range(len(mquery_label)):
        mquery_index1 = np.argwhere(mquery_label == mquery_label[i])  # 返回非0元素的索引
        mq = torch.mean(mquery_feature[mquery_index1, :], dim=0)

        query_path = query_paths[i]
        ap_tmp, CMC_tmp = evaluate(mq, mquery_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        # CMC = CMC + CMC_tmp
        # ap += ap_tmp
        if CMC_tmp[0] == 0:
            query_path = query_paths[i]
            query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
            orb = cv2.ORB_create()
            kp_q, des_q = orb.detectAndCompute(query_img, None)
            if des_q is None:
                continue
            scores = []
            for img_path in gallery_paths:
                des_g = gallery_orb_features[img_path]
                score = orb_match(des_q, des_g)
                scores.append((img_path, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            top5_img_paths = [item[0] for item in scores[:5]]

            top5_indices = [gallery_paths.index(path) for path in top5_img_paths]
            rerank_gallery_feature = gallery_feature[top5_indices]
            rerank_gallery_label = gallery_label[top5_indices]
            ap_tmp, CMC_tmp = evaluate(mq, mquery_label[i], rerank_gallery_feature, rerank_gallery_label)
        if len(CMC_tmp) < len(CMC):
            CMC_tmp_1 = torch.cat([CMC_tmp, torch.IntTensor([0]*(len(CMC)-len(CMC_tmp)))]).zero_()
        else:
            CMC_tmp_1 = CMC_tmp[:len(CMC)]
        CMC1 = CMC1 + CMC_tmp_1
        ap1 = ap1 + ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(mquery_label)  # average CMC
    CMC1 = CMC1.float()
    CMC1 = CMC1 / len(mquery_label)  # average CMC
    # print('multi Rank@1:%.2f Rank@5:%.2f Rank@10:%.2f mAP:%.2f' % (CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, ap / len(mquery_label) * 100))
    print('multi Rank@1:%.2f Rank@5:%.2f Rank@10:%.2f mAP:%.2f' % (CMC1[0] * 100, CMC1[4] * 100, CMC1[9] * 100, ap1 / len(mquery_label) * 100))

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


