#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re

def py_list_convert(value):
    """
    Change the list string into list
    """
    ans = []
    if isinstance(value, list):
        ans = value
    elif isinstance(value, int):
        ans.append(value)
    elif isinstance(value, float):
        ans.append(value)
    elif isinstance(value, str):
        ans = list(re.split(r'[,\s]\s*', value))
        ans = list(map(int, ans))
    return ans

if __name__ == "__main__":
    print("Unit test")
    a = py_list_convert(5)
    print(a)
    a = py_list_convert([1, 2, 3])
    print(a)
    a = py_list_convert('7 8 9 10')
    print(a)
    a = py_list_convert('2, 3, 4, 5')
    print(a)
