#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by youshaox on 17/1/19
"""
function:

"""

def print_object_debug(args, type):
    """
    :param args:
    :return:
    """
    if type == "functoin":
        print("-------------------------------- 开始运行function:\t {} --------------------------------".format(args))
    elif type == "object":
        print("-------------------------------- 开始运行object:\t {} --------------------------------".format(args))
        # print("-------------------------------- object的attributes有:\t {} --------------------------------".format(args.__dict__))
        # print("-------------------------------- 运行object结束:\t {} --------------------------------".format(args))
