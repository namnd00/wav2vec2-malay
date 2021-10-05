# # coding:utf-8
# """
# Name : generate_train_config_yaml.py
# Author : Nam Nguyen
# Contact : nam.nd.d3@gmail.com
# Time    : 9/28/2021 11:57 AM
# Desc:
# """
# import yaml
# from transformers import HfArgumentParser
#
# from argument_classes import DataTrainingArguments, ParameterArguments
#
# parser = HfArgumentParser((DataTrainingArguments, ParameterArguments))
# data_args, param_args = parser.parse_args_into_dataclasses()
#
# with open("sweep.yaml", "r") as stream:
#     try:
#         f = yaml.safe_load(stream)
#         # dict_keys(['program', 'project', 'parameters', 'command'])
#         # for key in f.keys():
#         #     print(f[key])
#     except yaml.YAMLError as exc:
#         print(exc)
#
# for arg in :
#     print(arg)
