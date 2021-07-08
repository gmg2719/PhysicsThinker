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

#ifndef _TEMP_MATRIX_H_
#define _TEMP_MATRIX_H_         1

#include "geometry/Grid.h"
#include "geometry/Field.h"
#include "geometry/Boundary.h"

class TempMatrix
{
private:
    Grid_cfd& grid_;
    Field_cfd& field_;
    Boundary_cfd& bd_;
    std::map<int, double> a_w, a_e, a_p;
    std::map<int, double> s_u;

    std::vector<double> lowerDiagonal;
    std::vector<double> diagonal;
    std::vector<double> upperDiagonal;

public:
    TempMatrix(Grid_cfd& grid_, Field_cfd& field);
    ~TempMatrix() {}

    void divergence(const double& constant, const Field_cfd& variable);

    void laplacian(const double& constant, const Field_cfd& variable);

    void solve();

    void tdma();
};

#endif
