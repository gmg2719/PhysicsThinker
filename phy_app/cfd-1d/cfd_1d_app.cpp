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
#include <ostream>
#include <fstream>
#include "inputOutput.h"
#include "TempMatrix.h"
#include "geometry/Grid.h"
#include "geometry/Field.h"

using namespace std;

int main(void)
{
    inputOutput io;

    // Print information
    io.print_header(std::cout);

    // Create the grid
    Grid_cfd grid;
    // Create the field
    Field_cfd field(grid);

    TempMatrix temp_matrix(grid, field);

    // Diffusion coefficient
    double k = 0.001;
    double coeff = 1.0;

    temp_matrix.laplacian(k, field);
    temp_matrix.divergence(coeff, field);

    // Solve the temperature matrix
    temp_matrix.solve();

    // Output the results
    io.print_results(std::cout, grid, field);

    std::ofstream file("results.txt");
    io.print_results(file, grid, field);
    file.close();

    io.print_time();

    return 0;
}

