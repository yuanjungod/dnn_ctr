# -*- coding:utf-8 -*-

from utils import data_preprocess
from model import DeepFM
import torch

result_dict = data_preprocess.read_criteo_data('./data/tiny_train_input.csv', './data/category_emb.csv')
test_dict = data_preprocess.read_criteo_data('./data/tiny_test_input.csv', './data/category_emb.csv')

print(result_dict['index'][0])
print(result_dict['value'][0])
print(result_dict['feature_sizes'])
# exit()

# torch.set_num_threads(8)
with torch.cuda.device(0):
# deepfm = DeepFM.DeepFM(39, result_dict['feature_sizes'], verbose=True, use_cuda=True, weight_decay=0.0001,
#                        use_fm=True, use_ffm=False, use_deep=True).cuda()
    deepfm = DeepFM.DeepFM(39, result_dict['feature_sizes'], verbose=True, use_cuda=True, weight_decay=0.0001,
                           use_fm=True, use_ffm=False, use_deep=True)
    deepfm.fit(result_dict['index'], result_dict['value'], result_dict['label'],
               test_dict['index'], test_dict['value'], test_dict['label'], ealry_stopping=True, refit=True)
