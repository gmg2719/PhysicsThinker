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

#ifndef _BOUNDARY_HPP_
#define _BOUNDARY_HPP_            1

#include "Grid.h"
#include "Field.h"

class Boundary_cfd
{
private:
    Grid_cfd& grid_;
    std::map<int, Cell_cfd> sp_;
public:
    Boundary_cfd(Grid_cfd& grid);
    ~Boundary_cfd() {}
    void apply_convection(std::map<int, double>& a_w, std::map<int, double>& a_p, std::map<int, double>& a_e,
                          const double& constant, const Field_cfd& variable);
    void apply_diffusion(std::map<int, double>& a_w, std::map<int, double>& a_p, std::map<int, double>& a_e,
                          const double& constant, const Field_cfd& variable);
};

#endif
