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
#include "common/log.h"
#include "geometry/Surface.h"
#include "geometry/Cell.h"
#include "geometry/Universe.h"
#include "geometry/Mesh.h"
#include "geometry/Material.h"
#include "geometry/Geometry.h"
#include "geometry/TrackGenerator.h"
#include "CPUSolver.h"

double uo2_total_xs[7] = {1.779490E-01, 3.298050E-01, 4.803880E-01, 5.543670E-01, 3.118010E-01, 3.951680E-01,  5.644060E-01};
double uo2_abs_xs[7] = {8.024800E-03, 3.717400E-03, 2.676900E-02, 9.623600E-02, 3.002000E-02, 1.112600E-01, 2.827800E-01};
double uo2_scatter_xs[49] = {1.275370E-01, 4.237800E-02, 9.437400E-06, 5.516300E-09, 0., 0., 0.,
                         0., 3.244560E-01, 1.631400E-03, 3.142700E-09, 0., 0., 0.,
                         0., 0., 4.509400E-01, 2.679200E-03, 0., 0., 0., 
                         0., 0., 0., 4.525650E-01, 5.566400E-03, 0., 0.,
                         0., 0., 0., 1.252500E-04, 2.714010E-01, 1.025500E-02, 1.002100E-08,
                         0., 0., 0., 0., 1.296800E-03, 2.658020E-01, 1.680900E-02, 
                         0., 0., 0., 0., 0., 8.545800E-03, 2.730800E-01};
double uo2_fis_xs [7]= {7.212060E-03, 8.193010E-04, 6.453200E-03, 1.856480E-02, 1.780840E-02, 8.303480E-02, 2.160040E-01};
double uo2_nufis_xs[7] = {2.005998E-02, 2.027303E-03, 1.570599E-02, 4.518301E-02, 4.334208E-02, 2.020901E-01, 5.257105E-01};
double uo2_chi [7]= {5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0., 0., 0.};

double mox43_total_xs[7] = {1.787310E-01, 3.308490E-01, 4.837720E-01, 5.669220E-01, 4.262270E-01, 6.789970E-01, 6.828520E-01};
double mox43_abs_xs[7] = {8.433900E-03, 3.757700E-03, 2.797000E-02, 1.042100E-01, 1.399400E-01, 4.091800E-01, 4.093500E-01};
double mox43_scatter_xs[49] = {1.288760E-01, 4.141300E-02, 8.229000E-06, 5.040500E-09, 0., 0., 0.,
                           0., 3.254520E-01, 1.639500E-03, 1.598200E-09, 0., 0., 0.,
                           0., 0., 4.531880E-01, 2.614200E-03, 0., 0., 0.,
                           0., 0., 0., 4.571730E-01, 5.539400E-03, 0., 0.,
                           0., 0., 0., 1.604600E-04, 2.768140E-01, 9.312700E-03, 9.165600E-09,
                           0., 0., 0., 0., 2.005100E-03, 2.529620E-01, 1.485000E-02,
                           0., 0., 0., 0., 0., 8.494800E-03, 2.650070E-01};
double mox43_fis_xs[7] = {7.62704E-03, 8.76898E-04, 5.69835E-03, 2.28872E-02, 1.07635E-02, 2.32757E-01, 2.48968E-01};
double mox43_nufis_xs[7] = {2.175300E-02, 2.535103E-03, 1.626799E-02, 6.547410E-02, 3.072409E-02, 6.666510E-01, 7.139904E-01};
double mox43_chi[7] = {5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0., 0., 0.};

double water_total_xs[7] = {6.010500E-04, 1.579300E-05, 3.371600E-04, 1.940600E-03, 5.741600E-03, 1.500100E-02, 3.723900E-02};
double water_abs_xs[7] = {1.592060E-01, 4.129700E-01, 5.903100E-01, 5.843500E-01, 7.180000E-01, 1.254450E+00, 2.650380E+00};
double water_scatter_xs[49] = {4.447770E-02, 1.134000E-01, 7.234700E-04, 3.749900E-06, 5.318400E-08, 0., 0.,
                           0., 2.823340E-01, 1.299400E-01, 6.234000E-04, 4.800200E-05, 7.448600E-06, 1.045500E-06,
                           0., 0., 3.452560E-01, 2.245700E-01, 1.699900E-02, 2.644300E-03, 5.034400E-04,
                           0., 0., 0., 9.102840E-02, 4.155100E-01, 6.373200E-02, 1.213900E-02,
                           0., 0., 0., 7.143700E-05, 1.391380E-01, 5.118200E-01, 6.122900E-02,
                           0., 0., 0., 0., 2.215700E-03, 6.999130E-01, 5.373200E-01,
                           0., 0., 0., 0., 0., 1.324400E-01, 2.480700E+00};
double water_fis_xs[7] = {0., 0., 0., 0., 0., 0., 0.};
double water_nufis_xs[7] = {0., 0., 0., 0., 0., 0., 0.};
double water_chi[7] = {0., 0., 0., 0., 0., 0., 0.};

void tiny_lattice_demo()
{
    std::cout << "Set the main simulation parameters ..." << std::endl;
    int num_threads = 1;
    double track_spacing = 0.1;
    int num_azim = 4;
    double tolerance = 1.0E-5;
    int max_iters = 200;
    set_log_level("NORMAL_LOG");
    std::cout << "Create the materials ..." << std::endl;
    Material uo2(1);
    Material water(2);
    uo2.setNumEnergyGroups(7);
    uo2.setSigmaT(uo2_total_xs, 7);
    uo2.setSigmaA(uo2_abs_xs, 7);
    uo2.setSigmaS(uo2_scatter_xs, 49);
    uo2.setSigmaF(uo2_fis_xs, 7);
    uo2.setNuSigmaF(uo2_nufis_xs, 7);
    uo2.setChi(uo2_chi, 7);
    water.setNumEnergyGroups(7);
    water.setSigmaT(water_total_xs, 7);
    water.setSigmaA(water_abs_xs, 7);
    water.setSigmaS(water_scatter_xs, 49);
    water.setSigmaF(water_fis_xs, 7);
    water.setNuSigmaF(water_nufis_xs, 7);
    water.setChi(water_chi, 7);
    std::cout << "Create surfaces ..." << std::endl;
    Circle circle(0.0, 0.0, 0.8);
    XPlane left(-2.0);
    XPlane right(2.0);
    YPlane top(2.0);
    YPlane bottom(-2.0);
    left.setBoundaryType(REFLECTIVE);
    right.setBoundaryType(REFLECTIVE);
    top.setBoundaryType(REFLECTIVE);
    bottom.setBoundaryType(REFLECTIVE);
    std::cout << "Create cells ..." << std::endl;
    Universe u0(0);
    Universe u1(1);
    Cell *cell0 = new CellBasic(1, 1);
    Cell *cell1 = new CellBasic(1, 2);
    Cell *cell2 = new CellFill(0, 2);
    cell0->addSurface(-1, &circle);
    cell1->addSurface(1, &circle);
    cell2->addSurface(1, &left);
    cell2->addSurface(-1, &right);
    cell2->addSurface(1, &bottom);
    cell2->addSurface(-1, &top);
    std::cout << "Create lattices ..." << std::endl;
    Lattice lattice(2, 2.0, 2.0);
    int universe_id[4] = {1, 1, 1, 1};
    lattice.setLatticeCells(2, 2, universe_id);
    std::cout << "Create the geometry ..." << std::endl;
    Geometry geom;
    geom.addMaterial(&uo2);
    geom.addMaterial(&water);
    geom.addCell(cell0);
    geom.addCell(cell1);
    geom.addCell(cell2);
    geom.addLattice(&lattice);
    geom.initializeFlatSourceRegions();
    std::cout << "Create the track generator ..." << std::endl;
    TrackGenerator tracks(&geom, num_azim, track_spacing);
    tracks.generateTracks();

    std::cout << "start to run the simulation ..." << std::endl;
    CPUSolver moc_solver(&geom, &tracks);
    moc_solver.setNumThreads(num_threads);
    moc_solver.setSourceConvergenceThreshold(tolerance);
    moc_solver.convergeSource(max_iters);
    moc_solver.printTimerReport();
}

void simple_lattice_demo()
{
}

void pin_lattice_demo()
{
}

void nested_lattice_demo()
{
}

void large_lattice_demo()
{
}

void full_core_demo()
{
}

void bundled_lattice_demo()
{
}

int main(int argc, char *argv[])
{
    std::cout << "Run openmoc 2D neutron transport simulation !" << std::endl;

    tiny_lattice_demo();
    simple_lattice_demo();
    pin_lattice_demo();
    nested_lattice_demo();
    large_lattice_demo();
    full_core_demo();
    bundled_lattice_demo();
    
    std::cout << "Finished !" << std::endl;

    return 0;
}
