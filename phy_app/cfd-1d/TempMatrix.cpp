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

#include <iostream>
#include <utility>
#include <map>
#include "TempMatrix.h"

TempMatrix::TempMatrix(Grid_cfd& grid, Field_cfd& fld)
{
    grid_ = &grid;
    fld_ = &fld;
    bd_ = new Boundary_cfd(grid);

    for (int i = 1; i <= grid_->ncells(); ++i) {
        s_u.insert(std::make_pair<int, double>((int)i, (double)0.0));
    }

    for (int i = 1; i <= grid_->ncells(); ++i) {
        a_w.insert(std::make_pair<int, double>((int)i, (double)0.0));
        a_e.insert(std::make_pair<int, double>((int)i, (double)0.0));
    }

    for (int i = 1; i < grid_->ncells(); ++i) {
        a_p.insert(std::make_pair<int, double>((int)i, (double)0.0));
    }
}

void TempMatrix::divergence(const double& constant, const Field_cfd& variable)
{
    for (int i=1; i<=grid_->ncells(); ++i) {
        a_w[i] += std::max(0.0, constant);
        a_e[i] += std::max(0.0, -constant);
    }

    a_w[1] = 0.0;
    a_e[grid_->ncells()] = 0.0;

    s_u[1] += constant*variable[0];

    bd_->apply_convection(a_w, a_p, a_e, constant, variable);
}

void TempMatrix::laplacian(const double& constant, const Field_cfd& variable)
{
    for (int i=1; i<=grid_->ncells(); ++i) {
        a_w[i] += constant*grid_->get_cell(i).A()/grid_->get_cell(i).dx();
        a_e[i] += constant*grid_->get_cell(i).A()/grid_->get_cell(i).dx();
    }

    a_w[1] = 0.0;
    a_e[grid_->ncells()] = 0.0;

    s_u[1] += 2*variable[0]*constant*grid_->get_cell(1).A()/grid_->get_cell(1).dx();    
    s_u[grid_->ncells()] += 2*variable[grid_->ncells()+1]*constant*grid_->get_cell(grid_->ncells()).A()/grid_->get_cell(grid_->ncells()).dx(); 

    bd_->apply_diffusion(a_w, a_p, a_e, constant, variable);
}

void TempMatrix::solve()
{
    for (int i=1; i<=grid_->ncells(); ++i) {
        a_p[i] += (a_w[i] + a_e[i]);
    }

    tdma();
}

void TempMatrix::tdma()
{
    std::vector<double> A, C;

    A.push_back(0.0);
    C.push_back(0.0);

    for (int i=1; i<=grid_->ncells(); ++i) {    
        A.push_back(a_e[i]/(a_p[i]-a_w[i]*A[i-1]));
        C.push_back((a_w[i]*C[i-1] + s_u[i])/(a_p[i] - a_w[i]*A[i-1]));
    }

    for (int i=grid_->ncells(); i>=1; --i)
    {
        (*fld_)[i] = A[i] * (*fld_)[i+1] + C[i];
    }
}

