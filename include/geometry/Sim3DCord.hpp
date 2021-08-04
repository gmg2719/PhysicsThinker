// MIT License
// 
// Copyright (c) 2021 PingzhouMing
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _SIM_3D_CORD_H_
#define _SIM_3D_CORD_H_         1

#include <cstdio>
#include <cstdlib>
#include <cmath>

class Sim3DCord
{
public:
    double x_;
    double y_;
    double z_;
public:
    Sim3DCord(double x, double y, double z=0.) : x_(x), y_(y), z_(z) {}

    double calc_distance(Sim3DCord& pos)
    {
        return sqrt((x_-pos.x_)*(x_-pos.x_) + (y_-pos.y_)*(y_-pos.y_) + (z_-pos.z_)*(z_-pos.z_));
    }

    void debug_print()
    {
        printf("Coordinate is : (%.6f, %.6f, %.6f) \n", x_, y_, z_);
    }
};

#endif
