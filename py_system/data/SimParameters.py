#!/usr/bin/python3
# -*- coding: utf-8 -*-

class SimParameters(object):
    def __init__(self):
        self.run_time = 0
        self.name = "default"

    def debug(self):
        print('SimParameters instance :')
        print('---------------------------------------------------')
