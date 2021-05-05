import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import jaccard_score
from sklearn import tree
import os
import time

import sys
sys.path.append('..')
from util import multi_label_metric

model_name = 'ECC'

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

def create_dataset(data, diag_voc, pro_voc, med_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)


def main():
    # grid_search = False
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    epoch = 100

    np.random.seed(epoch)
    np.random.shuffle(data)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point+eval_len:]
    data_test = data[split_point:split_point + eval_len]

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    test_X, test_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, pro_voc, med_voc)

    # confirmed_index = np.where(train_y.sum(axis=0) == 0)[0]
    # predicted_index = np.where(train_y.sum(axis=0) > 0)[0]
    # train_X, train_y = train_X[predicted_index], train_y[predicted_index]

    base_dt = LogisticRegression()

    tic_total_fit = time.time()
    global chains
    chains = [ClassifierChain(base_dt, order='random', random_state=i) for i in range(10)]
    for i, chain in enumerate(chains):
        tic = time.time()
        chain.fit(train_X, train_y)
        fittime = time.time() - tic
        print ('id {}, fitting time: {}'.format(i, fittime))
    print ('total fitting time: {}'.format(time.time() - tic_total_fit))

    # exit()

    tic = time.time()
    y_pred_chains = np.array([chain.predict(test_X) for chain in chains])
    y_prob_chains = np.array([chain.predict_proba(test_X) for chain in chains])
    pretime = time.time() - tic
    print ('inference time: {}'.format(pretime))

    y_pred = y_pred_chains.mean(axis=0)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    y_prob = y_prob_chains.mean(axis=0)

    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y, y_pred, y_prob)

    # ddi rate
    ddi_A = dill.load(open('../data/ddi_A_final.pkl', 'rb'))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm==1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    ddi_rate = dd_cnt / all_cnt
    print('Epoch: {}, DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        epoch, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, med_cnt / visit_cnt
        ))

if __name__ == '__main__':
    main()   
