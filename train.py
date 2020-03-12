import tensorflow as tf
from DataIterator import DataIterator
from Model import Model_DIN_V2_Gru_Vec_attGru_Neg
from tensorflow.python import debug as tfdbg
import numpy as np
import random

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2

best_auc = 0.0
import  os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

def train(train_file='data/train_data.csv',
          test_file='data/test_data.csv',
          app_id_path='info/app_id.csv',
          app_function_path='info/cut_app_function.csv',
          dictionary_path = 'info/dictionary.dict',
          batch_size=1,
          test_iter=1,
          save_iter=1,
          model_type='DNN',
          seed=2
          ):
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)

    with tf.Session() as sess:
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        train_data = DataIterator(train_file,batch_size,app_id_path,app_function_path,dictionary_path,neg_return=True)
        test_data = DataIterator(test_file,batch_size,app_id_path,app_function_path,dictionary_path,neg_return=True)
        (nums_name_ids,nums_function_ids,nums_tracks) = train_data.get_n()

        model = Model_DIN_V2_Gru_Vec_attGru_Neg(nums_name_ids, nums_function_ids,nums_tracks,
                                                EMBEDDING_DIM, HIDDEN_SIZE,ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        iter = 0
        lr = 0.001
        epoch_nums = 4
        for epoch in range(0, epoch_nums):
            loss_sum = 0.0
            accuracy_sum = 0.0
            aux_loss_sum = 0.0
            for sample_batch in train_data.batch_yield():
                (target_batch, nameid_batch, funcid_batch, nameid_his_batch,
                 funcid_his_batch, trackid_his_batch, sl_batch, mask, neg_funcid_his_batch,
                 neg_trackid_his_batch) = train_data.prepare_data(sample_batch)
                loss, accuracy, aux_loss = model.train(sess, [target_batch,nameid_batch,funcid_batch,
                                   nameid_his_batch,funcid_his_batch,
                                   trackid_his_batch, lr, mask, sl_batch,
                                   neg_funcid_his_batch,neg_trackid_his_batch],iter + 1)
                loss_sum += loss
                accuracy_sum += accuracy
                aux_loss_sum += aux_loss
                iter += 1

                if (iter % test_iter) == 0:
                    print("iter: %d ----> train_loss: %.8f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f" % \
                          (iter, loss_sum/test_iter, accuracy_sum/test_iter, aux_loss_sum/test_iter))

                    print("test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss:%.4f" %\
                          eval(sess,test_data, model, best_model_path))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' % (iter))
                    model.save(sess, model_path + "--" + str(iter))
            lr *= 0.5

def eval(sess, test_data, model, model_path):

    loss_sum = 0.0
    accuracy_sum = 0.0
    aux_loss_sum = 0.0

    nums = 0
    cat_stored_attr = {}
    for c in range(0, 11):
        cat_stored_attr[c] = []
    for sample_batch in test_data.batch_yield():
        nums += 1
        (target_batch, nameid_batch, funcid_batch, nameid_his_batch,
         funcid_his_batch, trackid_his_batch, sl_batch, mask, neg_funcid_his_batch,
         neg_trackid_his_batch) = test_data.prepare_data(sample_batch)

        probs, loss, accuracy, aux_loss = model.calculate(sess, [target_batch,nameid_batch,funcid_batch,
                                                                 nameid_his_batch,funcid_his_batch,
                                                                 trackid_his_batch, mask,sl_batch,
                                                                 neg_funcid_his_batch, neg_trackid_his_batch])

        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += accuracy

        prob_maxind = probs.argmax(1).tolist()

        target_maxind = target_batch.argmax(1).tolist()
        n_sample = len(prob_maxind)
        # print('target={}'.format(target_batch))
        # print('prob={}'.format(probs))
        # print('prob_maxid={}'.format(prob_maxind))
        # print('target_maxind={}'.format(target_maxind))

        for category in range(0, 11):
            for ind in range(n_sample):
                if prob_maxind[ind] == category:
                    p = 1.0
                else:
                    p = 0.0

                if target_maxind[ind] == category:
                    t = 1.0
                else:
                    t = 0.0
                cat_stored_attr[category].append([p, t])

    # print('cat_stored_attr={}'.format(cat_stored_attr))
    cat_auc_score = []
    for c in range(0, 11):
        auc_score = calc_auc(cat_stored_attr[c])
        cat_auc_score.append(auc_score)

    accuracy_sum = accuracy_sum/nums
    loss_sum = loss_sum/nums
    aux_loss_sum /nums

    test_auc = max(cat_auc_score)

    global best_auc

    if test_auc > best_auc:
        best_auc = test_auc
        model.save(sess, model_path)

    return test_auc, loss_sum, accuracy_sum, aux_loss_sum



def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    if pos == 0 or neg == 0:
        return 0.0

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc

if __name__ == '__main__':
    SEED = 5
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    train(batch_size=128,test_iter=10,save_iter=50)





