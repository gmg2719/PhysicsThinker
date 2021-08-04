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

#ifndef _POSITIONING_5G_H_
#define _POSITIONING_5G_H_          1

#include "my_physics_constants.h"
#include "geometry/Sim3DCord.hpp"

#define TAYLOR_DIRECT_METHOD        1
#define NEWTON_ITER_METHOD          2
#define NEWTON_FREE_ITER_METHOD     3
#define FIXED_ITER_METHOD           4
#define NEWTON_ITER_MAXTIME         15

Sim3DCord& tdoa_positioning_4bs(int type, Sim3DCord& bs1, Sim3DCord& bs2, Sim3DCord& bs3, Sim3DCord& bs4,
                                double dt21, double dt31, double dt41);

#endif
