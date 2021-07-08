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

#ifndef _FIELD_HPP_
#define _FIELD_HPP_           1

#include "Grid.h"

class Field_cfd
{
    const Grid_cfd& grid_;
    std::vector<double> data_;
public:
    Field_cfd(const Grid_cfd& grid) : grid_(grid)
    {
        data_ = std::vector<double>(grid_.ncells()+2);
        data_[0] = 1;
        data_[grid_.ncells() + 1] = 2;
    }
    ~Field_cfd() {}
    double operator [](int i) const
    {
        return data_[i];
    }
    double& operator [](int i)
    {
        return data_[i];
    }
};

#endif
