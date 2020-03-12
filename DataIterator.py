#--------------------------------------------------
# DataIterator
#描述：定义样本的生成器。
# 从训练数据或测试数据中读出样本，
# 结合app_id.csv和样本中的app名称字段，将app名称转换成相应的id;
# 结合app_function.csv和样本中的app周数字段，先找到app在该周数对应的功能，然后将app功能转换成相应的id;
# 样本中的轨道值字段可直接作为id返回；
# 负采样：在历史周数的每一周，采集与历史app不同的其他app的功能

from Utils import read_csv
import os, csv
import pickle
import random
import numpy as np
class DataIterator:

    data_path = None
    batch_size = None

    name_id = None
    name_week_function = None

    dictionary = None

    neg_return = False
    nums_name_ids = 0
    nums_function_ids = 0
    nums_tracks = 11
    seq_len = 6
    neg_nums = 2


    def __init__(self,data_path,
                 batch_size,
                 app_id_path,
                 app_function_path,
                 dictionary_path,
                 neg_return):

        self.data_path = data_path
        self.batch_size = batch_size

        self.name_id = read_csv(app_id_path, 2)
        self.name_week_function = read_csv(app_function_path, 2)

        self.dictionary = pickle.load(open(dictionary_path, 'rb'))


        self.nums_function_ids = len(self.dictionary)
        self.nums_name_ids = len(self.name_id)
        self.neg_return = neg_return


    def batch_yield(self):
        samples_parse = []
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as fp:
                reader = csv.reader(fp)

                for sample in reader:
                    sample_result = self.sample_parse(sample)
                    samples_parse.append(sample_result)

                    if len(samples_parse) == self.batch_size:
                        yield samples_parse
                        samples_parse.clear()
                if len(samples_parse) != 0:
                    yield samples_parse



    def sample_parse(self,sample):

        track = sample[0]
        name = sample[1]
        week = sample[2]

        name_his = sample[3]
        week_his = sample[4:6]
        track_his = sample[6:]


        target = [0.] * self.nums_tracks
        target[int(track) - 1] = 1

        nameid = self.name2id(name)
        funcid = self.function2id(name, week)
        nameid_his = self.name2id(name_his)
        funcid_his = self.function2id(name_his, week_his)
        trackid_his = [int(t) for t in track_his]


        if self.neg_return:
            (neg_funcid_his, neg_trackid_his) = self.negsample_generate(name_his, week_his, track_his)
            return (target, nameid, funcid, nameid_his, funcid_his, trackid_his, neg_funcid_his, neg_trackid_his)
        else:
            return(target, nameid, funcid, nameid_his, funcid_his, trackid_his)





    def name2id(self, appname):
        id_result = None
        filter_results = [line[1] for line in self.name_id if line[0] == appname]
        if len(filter_results) != 0:
            id_result = filter_results[0]
        return int(id_result)


    def function2id(self,appname, weeks):

        filter_results = [line for line in self.name_week_function if line[0] == appname]

        ids_result = []
        if type(weeks) == list:
            start = int(weeks[0])
            end = int(weeks[1])
            for week in range(start, end + 1):
                week_function = [line[2:] for line in filter_results if line[1] == str(week)]
                function_id = [self.dictionary[function_word] for function_word in week_function[0] if function_word in self.dictionary.keys()]
                ids_result.append(function_id)
        else:
            week_function = [line[2:] for line in filter_results if line[1] == weeks]
            function_id = [self.dictionary[function_word] for function_word in week_function[0] if function_word in self.dictionary.keys()]
            ids_result.extend(function_id)

        return ids_result


    def negsample_generate(self, appname,weeks,tracks):


        neg_ids_function = []
        neg_ids_track = []


        start = int(weeks[0])
        end = int(weeks[1])

        for week in range(start, end + 1):
            week_function = [line for line in self.name_week_function if line[0] != appname and line[1] == str(week)]
            neg_index = random.sample(range(0, len(week_function)), self.neg_nums)

            neg_ids_function_week = []
            for ind in neg_index:
                neg_ids_function_week.append([self.dictionary[word] for word in week_function[ind][2:] if word in self.dictionary])

            neg_ids_function.append(neg_ids_function_week)

            can_selected_track = [t for t in range(1, self.nums_tracks + 1)]
            can_selected_track.remove(int(tracks[week - start]))


            neg_ids_track_week = random.sample(can_selected_track, self.neg_nums)
            neg_ids_track.append(neg_ids_track_week)

        return (neg_ids_function,neg_ids_track)

    def get_n(self):
        return (self.nums_name_ids, self.nums_function_ids, self.nums_tracks)

    def prepare_data(self, sample_parse):

        target_batch = np.array([sample[0] for sample in sample_parse]) - 1
        nameid_batch = np.array([sample[1] for sample in sample_parse])

        funcid_batch_org = [sample[2] for sample in sample_parse]

        nameid_his_batch = np.array([sample[3] for sample in sample_parse])

        funcid_his_batch_org = [sample[4] for sample in sample_parse]
        trackid_his_batch_org = [sample[5]for sample in sample_parse]

        neg_funcid_his_batch_org = [sample[6] for sample in sample_parse]
        neg_trackid_his_batch_org = [sample[7] for sample in sample_parse]

        len_words = [len(item[2:]) for item in self.name_week_function]

        n_samples = len(sample_parse)
        seq_len = self.seq_len
        max_words = max(len_words)
        neg_samples = self.neg_nums


        funcid_batch = np.zeros((n_samples, max_words)).astype("int64")
        funcid_his_batch = np.zeros((n_samples, seq_len, max_words)).astype("int64")
        neg_funcid_his_batch = np.zeros((n_samples, seq_len, neg_samples, max_words)).astype("int64")

        trackid_his_batch = np.zeros((n_samples, seq_len)).astype("int64")
        neg_trackid_his_batch = np.zeros((n_samples, seq_len, neg_samples)).astype("int64")

        mask = np.zeros((n_samples, seq_len)).astype('float32')

        for i in range(0, len(funcid_batch_org)):
            funcid_batch[i, :len(funcid_batch_org[i])] = funcid_batch_org[i]
            for j in range(0, len(funcid_his_batch_org[i])):
                funcid_his_batch[i,j, :len(funcid_his_batch_org[i][j])] = funcid_his_batch_org[i][j]
                for k in range(0, len(neg_funcid_his_batch_org[i][j])):
                    neg_funcid_his_batch[i, j, k, :len(neg_funcid_his_batch_org[i][j][k])] = neg_funcid_his_batch_org[i][j][k]
            trackid_his_batch[i] = trackid_his_batch_org[i]
            trackid_his_batch[i] = trackid_his_batch[i] - 1
            mask[i, :] = 1.
            for p in range(0, len(neg_trackid_his_batch_org[i])):
                neg_trackid_his_batch[i, p, :len(neg_trackid_his_batch_org[i][p])] = neg_trackid_his_batch_org[i][p]
                neg_trackid_his_batch[i, p, :len(neg_trackid_his_batch_org[i][p])] -= 1


        sl_batch = [seq_len] * n_samples



        return (target_batch, nameid_batch, funcid_batch, nameid_his_batch,
                funcid_his_batch,trackid_his_batch, sl_batch, mask, neg_funcid_his_batch,
                neg_trackid_his_batch)










if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data', 'train_data.csv')
    app_id_path = os.path.join(os.getcwd(), 'info', 'app_id.csv')
    app_function_path = os.path.join(os.getcwd(), 'info', 'cut_app_function.csv')
    dictionary_path = os.path.join(os.getcwd(), 'info', 'dictionary.dict')
    iterator = DataIterator(data_path,128,app_id_path,app_function_path,dictionary_path,True)
    i = 1
    batches = iterator.batch_yield()
    for sample in batches:
        print("第{}batch样本：".format(i))
        (target_batch, nameid_batch, funcid_batch, nameid_his_batch,
         funcid_his_batch, trackid_his_batch, sl_batch, mask, neg_funcid_his_batch,
         neg_trackid_his_batch) = iterator.prepare_data(sample)

        i = i + 1

