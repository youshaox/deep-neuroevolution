#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by youshaox on 2/1/19
"""
function:

"""

class Student(object):
    # 加上@property，把一个get方法变为了属性
    @property
    def score(self):
        return self._score

    # @score.setter，负责把一个setter方法变成属性赋值
    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value

s = Student()
# OK，实际转化为s.set_score(60)
s.score = 60
# OK，实际转化为s.get_score()
print(s.score)