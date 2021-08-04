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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "my_utils.hpp"
#include "signal/positioning_5G.h"
#include "my_time.h"

int main(void)
{
    uint64_t start, end;
    uint64_t total_ticks = 0;
    Sim3DCord position(0.0, 0.0, 0.0);
    Sim3DCord bs1(1000, 1999, 6);
    Sim3DCord bs2(1, 1, 18);
    Sim3DCord bs3(1999, 1, 20);
    Sim3DCord bs4(500, 1000, 10);
    int counter = 0;

    for (int itr = 1; itr <= 2000; itr++)
    {
        Sim3DCord uav_est((double(rand()) / (double(RAND_MAX) + 1.0)) * 2000.0, (double(rand()) / (double(RAND_MAX) + 1.0)) * 2000.0,
                          (double(rand()) / (double(RAND_MAX) + 1.0)) * 200.0);
        printf("Outter %d for estimating the coordinate (%.4f, %.4f, %.4f)\n", itr, uav_est.x_, uav_est.y_, uav_est.z_);

        double r1 = uav_est.calc_distance(bs1);
        double r2 = uav_est.calc_distance(bs2);
        double r3 = uav_est.calc_distance(bs3);
        double r4 = uav_est.calc_distance(bs4);
        double dt21 = (r2 - r1) / MY_LIGHT_SPEED;
        double dt31 = (r3 - r1) / MY_LIGHT_SPEED;
        double dt41 = (r4 - r1) / MY_LIGHT_SPEED;

        start = my_us_gettimeofday();
        position = tdoa_positioning_4bs(NEWTON_ITER_METHOD, bs1, bs2, bs3, bs4, dt21, dt31, dt41);
        end = my_us_gettimeofday();
        total_ticks += (end - start);

        double err1 = std::max(std::max( fabs(position.x_ - uav_est.x_), fabs(position.y_ - uav_est.y_) ), fabs(position.z_ - uav_est.z_));
        if (err1 < 1.0) {
            counter += 1;
        }
    }

    printf("Average run time is %.6f us !\n", float(total_ticks)/2000);
    printf("Error < 1m CDF = %.4f \n", float(counter) / 2000);

    return 0;
}

