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

#ifndef _INPUT_OUTPUT_HPP_
#define _INPUT_OUTPUT_HPP_            1

#include <iostream>
#include <ostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include "geometry/Grid.h"

class inputOutput
{
    time_t start_time;
    time_t end_time;

public:
    inputOutput()
    {
        time(&start_time);
    }

    ~inputOutput() {}

    void print_header(std::ostream& os)
    {
        time_t rawtime;
        time(&rawtime);

        os << "\n|----------------------------------------------------|" << std::endl;
        os << "|simpleCFD: A 1D Convection-Diffusion Equation Solver|" << std::endl;
        os << "|----------------------------------------------------|" << std::endl;
        os << "|----------------Laurence R. McGlashan---------------|" << std::endl;
        os << "|--------------Modified by PingzhouMing--------------|" << std::endl;
        os << "|----------------------------------------------------|" << std::endl;    
        os << "|Time Tag: " << asctime(localtime(&rawtime));
        os << "|----------------------------------------------------|\n" << std::endl;
    }

    void print_results(std::ostream& os, Grid_cfd& grid, Field_cfd& field)
    {
        os << std::left << std::scientific << std::setw(10) << "# x (m)" << " " << "Temperature (K)" << std::endl;

        for (int i = 1; i <= grid.ncells(); ++i) {
            os << std::setw(10) << grid.get_cell(i).cell_centre() << " " << field[i] << std::endl;
        }
    }

    void print_time()
    {
        time(&end_time);
        std::cout << std::endl << "Simulation finished in : " << difftime(end_time, start_time) << " seconds !" << std::endl; 
    }
};

#endif
