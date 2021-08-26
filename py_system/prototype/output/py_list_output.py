#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

_NUMBER_OF_EACH_LINE = 8

def py_list_list2file(output_dir, file_name, var_list):
    """
    Output the 1D or 2D list to ASCII file.
    Args:
        output_dir : the directory of the output
        file_name  : output file name
        var_list   : 1D or 2D list
    """
    global _NUMBER_OF_EACH_LINE
    try:
        fd = open(output_dir + '/' + file_name, 'w', encoding='utf-8')
    except OSError:
        print('py_list_list2file() open file error !')
        return
    with fd:
        var_formatted = np.array(var_list)
        if var_formatted.ndim == 1:
            for item in var_formatted:
                fd.write(str(item) + '\n')
        elif var_formatted.ndim == 2:
            size = np.size(var_formatted)
            tmp = np.reshape(var_formatted, (size))
            line_counter = 0
            for item in tmp:
                line_counter += 1
                fd.write(str(item) + ' ')
                if np.mod(line_counter, _NUMBER_OF_EACH_LINE) == 0:
                    fd.write('\n')
        else:
            print('py_list_list2file() only supports 1D and 2D list !')
    fd.close()

def py_list_list2bin(output_dir, file_name, var_list):
    try:
        fd = open(output_dir + '/' + file_name, 'wb')
    except OSError:
        print('py_list_list2bin() open file error !')
        return
    with fd:
        var_formatted = np.array(var_list, dtype='float32')
        if var_formatted.ndim == 1:
            var_formatted.tofile(fd)
        elif var_formatted.ndim == 2:
            size = np.size(var_formatted)
            tmp = np.reshape(var_formatted, (size))
            tmp.tofile(fd)
        else:
            print('py_list_list2bin() only supports 1D and 2D list !')
    fd.close()

