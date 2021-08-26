#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np

class Sim3DBs(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def calc_distance(self, dest):
        return math.sqrt((self.x - dest.x)**2 + (self.y - dest.y)**2 + (self.z - dest.z)**2)
    def calc_distance(self, x, y, z):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)
    def debug_print(self):
        print('Coordinate is : %.6f, %.6f, %.6f' % (self.x, self.y, self.z))

class Sim3DUe(object):
    def __init__(self, x, y, z, v=0):
        self.x = x
        self.y = y
        self.z = z
        self.is_constant_speed = True
        self.vel = v
        # Support 14 directions mostly !
        # 0: x+, 1: x-, 2: y+, 3: y-, 4: z+, 5: z-
        # 6: x+y+, 7: x+y-, 8:x-y+, 9:x-y-
        # 10: y+z+, 11: y+z-, 12: y-z+, 13: y-z-
        # 14: x+z+, 15: x+z-, 16: x-z+, 17: x-z-
        # 18: x+y+z+, 19: x-y+z+, 20: x-y+z-, 21: x+y+z-
        # 22: x+y-z+, 23: x-y-z+, 24: x-y-z-, 25: x+y-z-
        self.direction = 0
    def calc_distance(self, dest):
        return math.sqrt((self.x - dest.x)**2 + (self.y - dest.y)**2 + (self.z - dest.z)**2)
    def calc_distance(self, x, y, z):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)
    def debug_print(self):
        print('Coordinate is : %.6f, %.6f, %.6f' % (self.x, self.y, self.z))

def compute_velocity_direction(vel, direct):
    velx = 0.
    vely = 0.
    velz = 0.
    if direct == 0:
        velx = vel
    elif direct == 1:
        velx = -vel
    elif direct == 2:
        vely = vel
    elif direct == 3:
        vely = -vel
    elif direct == 4:
        velz = vel
    elif direct == 5:
        velz = -vel
    elif direct == 6:
        velx = vel * (math.sqrt(2) / 2)
        vely = vel * (math.sqrt(2) / 2)
    elif direct == 7:
        velx = vel * (math.sqrt(2) / 2)
        vely = -vel * (math.sqrt(2) / 2)
    elif direct == 8:
        velx = -vel * (math.sqrt(2) / 2)
        vely = vel * (math.sqrt(2) / 2)
    elif direct == 9:
        velx = -vel * (math.sqrt(2) / 2)
        vely = -vel * (math.sqrt(2) / 2)
    elif direct == 10:
        vely = vel * (math.sqrt(2) / 2)
        velz = vel * (math.sqrt(2) / 2)
    elif direct == 11:
        vely = vel * (math.sqrt(2) / 2)
        velz = -vel * (math.sqrt(2) / 2)
    elif direct == 12:
        vely = -vel * (math.sqrt(2) / 2)
        velz = vel * (math.sqrt(2) / 2)
    elif direct == 13:
        vely = -vel * (math.sqrt(2) / 2)
        velz = -vel * (math.sqrt(2) / 2)
    elif direct == 14:
        velx = vel * (math.sqrt(2) / 2)
        velz = vel * (math.sqrt(2) / 2)
    elif direct == 15:
        velx = vel * (math.sqrt(2) / 2)
        velz = -vel * (math.sqrt(2) / 2)
    elif direct == 16:
        velx = -vel * (math.sqrt(2) / 2)
        velz = vel * (math.sqrt(2) / 2)
    elif direct == 17:
        velx = -vel * (math.sqrt(2) / 2)
        velz = -vel * (math.sqrt(2) / 2)
    # 18: x+y+z+, 19: x-y+z+, 20: x-y+z-, 21: x+y+z-
    # 22: x+y-z+, 23: x-y-z+, 24: x-y-z-, 25: x+y-z-
    elif direct == 18:
        velx = vel / math.sqrt(3)
        vely = vel / math.sqrt(3)
        velz = vel / math.sqrt(3)
    elif direct == 19:
        velx = -vel / math.sqrt(3)
        vely = vel / math.sqrt(3)
        velz = vel / math.sqrt(3)
    elif direct == 20:
        velx = -vel / math.sqrt(3)
        vely = vel / math.sqrt(3)
        velz = -vel / math.sqrt(3)
    elif direct == 21:
        velx = vel / math.sqrt(3)
        vely = vel / math.sqrt(3)
        velz = -vel / math.sqrt(3)
    elif direct == 22:
        velx = vel / math.sqrt(3)
        vely = -vel / math.sqrt(3)
        velz = vel / math.sqrt(3)
    elif direct == 23:
        velx = -vel / math.sqrt(3)
        vely = -vel / math.sqrt(3)
        velz = vel / math.sqrt(3)
    elif direct == 24:
        velx = -vel / math.sqrt(3)
        vely = -vel / math.sqrt(3)
        velz = -vel / math.sqrt(3)
    elif direct == 25:
        velx = vel / math.sqrt(3)
        vely = -vel / math.sqrt(3)
        velz = -vel / math.sqrt(3)
    else:
        print('Not supported discrete direction !')
        return 0, 0, 0
    return velx, vely, velz

def random_walk_3D_single(time_steps, ue):
    n = np.size(time_steps)
    if np.size(ue) != 1:
        print('random_walk_3D_single() only support 1 UE, you should use random_walk_3D_multiple() !')
        return np.array([]), np.array([]), np.array([])
    x_pos = np.zeros(n, dtype=float)
    y_pos = np.zeros(n, dtype=float)
    z_pos = np.zeros(n, dtype=float)

    if n <= 0:
        return np.array([]), np.array([]), np.array([])

    x_init = ue.x
    y_init = ue.y
    z_init = ue.z
    vel = ue.vel
    direct = np.random.randint(0, 26, size=n)
    for i in range(n):
        velx = 0.
        vely = 0.
        velz = 0.

        velx, vely, velz = compute_velocity_direction(vel, direct[i])
        
        if (i == 0):
            x_pos[0] = x_init + velx * time_steps[0]
            y_pos[0] = y_init + vely * time_steps[0]
            z_pos[0] = z_init + velz * time_steps[0]
        else:
            x_pos[i] = x_pos[i-1] + velx * (time_steps[i] - time_steps[i-1])
            y_pos[i] = y_pos[i-1] + vely * (time_steps[i] - time_steps[i-1])
            z_pos[i] = z_pos[i-1] + velz * (time_steps[i] - time_steps[i-1])

    return x_pos, y_pos, z_pos

def random_walk_3D_multiple(time_steps, ue_list):
    n = np.size(time_steps)
    num_ue = np.size(ue_list)
    x_pos = np.zeros((num_ue, n), dtype=float)
    y_pos = np.zeros((num_ue, n), dtype=float)
    z_pos = np.zeros((num_ue, n), dtype=float)

    if n <= 0:
        return np.array([]), np.array([]), np.array([])

    for k in range(num_ue):
        x_init = ue_list[k].x
        y_init = ue_list[k].y
        z_init = ue_list[k].z
        vel = ue_list[k].vel
        direct = np.random.randint(0, 26, size=n)
        for i in range(n):
            velx = 0.
            vely = 0.
            velz = 0.
    
            velx, vely, velz = compute_velocity_direction(vel, direct[i])
            
            if (i == 0):
                x_pos[k][0] = x_init + velx * time_steps[0]
                y_pos[k][0] = y_init + vely * time_steps[0]
                z_pos[k][0] = z_init + velz * time_steps[0]
            else:
                x_pos[k][i] = x_pos[k][i-1] + velx * (time_steps[i] - time_steps[i-1])
                y_pos[k][i] = y_pos[k][i-1] + vely * (time_steps[i] - time_steps[i-1])
                z_pos[k][i] = z_pos[k][i-1] + velz * (time_steps[i] - time_steps[i-1])

def ue_update_distance(bs, x_pos, y_pos, z_pos):
    if np.size(x_pos):
        return []
    update_dist = []
    for i in range(len(x_pos)):
        dist = bs.calc_distance(x_pos[i], y_pos[i], z_pos[i])
        update_dist.append(dist)
    return np.array(update_dist)

if __name__ == "__main__":
    print("Random walk test")
    ue = Sim3DUe(1.0, 1.0, 10.0, 3.0)
    steps = np.arange(0.5, 10.5, 0.5)
    x, y, z = random_walk_3D_single(steps, ue)
    bs1 = Sim3DBs(1000, 1900, 60)
    bs2 = Sim3DBs(1000, 1000, 80)
    bs3 = Sim3DBs(1900, 1000, 10)
    bs4 = Sim3DBs(500, 1000, 40)
