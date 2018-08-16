# -*- coding: utf-8 -*-
# @Time    : 18-8-10 下午7:57
# @Author  : zhangmr
# @File    : run.py

from zsl_demo.data_loader import SplitClassDataLoader
from zsl_demo.zsl import ZSL

if __name__ == "__main__":
    parameters = dict()
    parameters['data_dir'] = 'DatasetA/train'
    parameters['model_dir'] = 'output/encoder.pb'
    parameters['sigma'] = 0
    parameters['lambda'] = 1e-2
    parameters['gamma'] = 1e-2

    data_provider = SplitClassDataLoader()
    zsl = ZSL(data_provider, parameters)
    zsl.run()
