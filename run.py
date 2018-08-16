# -*- coding: utf-8 -*-
# @Time    : 18-8-16 下午4:47
# @Author  : zhangmr
# @File    : run.py
from generator import *
from SAE import SAE
import pandas as pd


def main():
    # train
    feature_extractor = FeatureExtractor("/home/zhangmr/project/my/zsl/teamwork/resource/simple_feature_extractor.pb")
    train_generator = ImgGenerator("/data/mydata/zeroshot/DatasetA_train_20180813", feature_extractor)
    train_generator.split_data()



    # test
    test_generator = ImgGenerator("/data/mydata/zeroshot/DatasetA_test_20180813", feature_extractor, is_train=False)



if __name__=='__main__':
    main()
