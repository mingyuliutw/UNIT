"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import yaml


class NetConfig(object):
    def __init__(self, config):
        stream = open(config,'r')
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                if k == "train":
                    for k1, v1 in v.items():
                        cmd = "self." + k1 + "=" + repr(v1)
                        print(cmd)
                        exec(cmd)
        stream.close()


def dict_from_class(cls):
    return dict(
        (key, value)
        for (key, value) in cls.__dict__.items())


class SettingConfig(object):
    def __init__(self, config):
        stream = open(config,'r')
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                if k == "train":
                    for k1, v1 in v.items():
                        cmd = "self." + k1 + "=" + repr(v1)
                        print(cmd)
                        exec(cmd)
        stream.close()
