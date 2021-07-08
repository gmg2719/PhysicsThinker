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

#ifndef _GRID_H_
#define _GRID_H_            1

#include <iostream>
#include <vector>
#include <map>
#include "Cell.h"

class Grid_cfd
{
private:
    std::map<int, Cell_cfd> cells_;
    int ncells_;
    double length_;
public:
    Grid_cfd() : ncells_(10000), length_(1.0)
    {
        for (int i = 1; i <= ncells_; ++i) {
            cells_.insert(std::pair<int, Cell_cfd>(i, Cell_cfd()));
        }

        for (int i = 1; i <= ncells_; ++i) {
            get_cell(i).set_centre((i-0.5)*length_/ncells_);
        }
    }
    ~Grid_cfd() {}
    bool load_grid() { return true; }
    inline const int& ncells() const { return ncells_; }
    inline Cell_cfd& get_cell(const int i) { return cells_[i]; }
};

#endif
