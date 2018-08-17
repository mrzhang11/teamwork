# -*- coding: utf-8 -*-
# @Time    : 18-8-16 下午4:47
# @Author  : zhangmr
# @File    : run.py
from generator import *
from SAE import SAE
from utils import *


def main():
    # train
    feature_extractor = FeatureExtractor("/home/zhangmr/project/my/zsl/teamwork/resource/simple_feature_extractor.pb")
    #train_generator = ImgGenerator("/data/mydata/zeroshot/DatasetA_train_20180813", feature_extractor)
    #params = dict()
    #params["lambdas"] = [0.9, 0.95, 1, 1.05, 1.1]

    #model = SAE(train_generator, params, load_weights=False)
    #model.train()

    # test
    test_generator = ImgGenerator("/data/mydata/zeroshot/DatasetA_test_20180813", feature_extractor, is_train=False)
    imgs = test_generator.x
    features = test_generator.xf
    model = SAE(test_generator, None, load_weights=True)
    predictions, _ = model.predict(features)
    submit(imgs, predictions,"/home/zhangmr/project/my/zsl/teamwork/resource/submit.txt")


if __name__=='__main__':
    main()
