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

//###############################################################################
//################################      UO2      ################################
//###############################################################################
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

//###############################################################################
//##############################      MOX (4.3%)     ############################
//###############################################################################
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

//###############################################################################
//##############################      MOX (7%)     ##############################
//###############################################################################
double mox7_total_xs[7] = {1.813230E-01, 3.343680E-01, 4.937850E-01, 5.912160E-01, 4.741980E-01, 8.336010E-01, 8.536030E-01};
double mox7_abs_xs[7] = {9.065700E-03, 4.296700E-03, 3.288100E-02, 1.220300E-01, 1.829800E-01, 5.684600E-01, 5.852100E-01};
double mox7_scatter_xs[49] = {1.304570E-01, 4.179200E-02, 8.510500E-06, 5.132900E-09, 0., 0., 0.,
                              0., 3.284280E-01, 1.643600E-03, 2.201700E-09, 0., 0., 0.,
                              0., 0., 4.583710E-01, 2.533100E-03, 0., 0., 0.,
                              0., 0., 0., 4.637090E-01, 5.476600E-03, 0., 0.,
                              0., 0., 0., 1.761900E-04, 2.823130E-01, 8.728900E-03, 9.001600E-09,
                              0., 0., 0., 0., 2.276000E-03, 2.497510E-01, 1.311400E-02,
                              0., 0., 0., 0., 0., 8.864500E-03, 2.595290E-01};
double mox7_fis_xs[7] = {8.25446E-03, 1.32565E-03, 8.42156E-03, 3.28730E-02, 1.59636E-02, 3.23794E-01, 3.62803E-01};
double mox7_nufis_xs[7] = {2.381395E-02, 3.858689E-03, 2.413400E-02, 9.436622E-02, 4.576988E-02, 9.281814E-01, 1.043200E+00};
double mox7_chi[7] = {5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0., 0., 0.};

//###############################################################################
//##############################      MOX (8.7%)     ############################
//###############################################################################
double mox87_total_xs[7] = {1.830450E-01, 3.367050E-01, 5.005070E-01, 6.061740E-01, 5.027540E-01, 9.210280E-01, 9.552310E-01};
double mox87_abs_xs[7] = {9.486200E-03, 4.655600E-03, 3.624000E-02, 1.327200E-01, 2.084000E-01, 6.587000E-01, 6.901700E-01};
double mox87_scatter_xs[49] = {1.315040E-01, 4.204600E-02, 8.697200E-06, 5.193800E-09, 0., 0., 0.,
                               0., 3.304030E-01, 1.646300E-03, 2.600600E-09, 0., 0., 0.,
                               0., 0., 4.617920E-01, 2.474900E-03, 0., 0., 0.,
                               0., 0., 0., 4.680210E-01, 5.433000E-03, 0., 0.,
                               0., 0., 0., 1.859700E-04, 2.857710E-01, 8.397300E-03, 8.928000E-09,
                               0., 0., 0., 0., 2.391600E-03, 2.476140E-01, 1.232200E-02,
                               0., 0., 0., 0., 0., 8.968100E-03, 2.560930E-01};
double mox87_fis_xs[7] = {8.67209E-03, 1.62426E-03, 1.02716E-02, 3.90447E-02, 1.92576E-02, 3.74888E-01, 4.30599E-01};
double mox87_nufis_xs[7] = {2.518600E-02, 4.739509E-03, 2.947805E-02, 1.122500E-01, 5.530301E-02, 1.074999E+00, 1.239298E+00};
double mox87_chi[7] = {5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0., 0., 0.};

//###############################################################################
//############################      Fission Chamber     #########################
//###############################################################################
double chamber_total_xs[7] = {1.260320E-01, 2.931600E-01, 2.842500E-01, 2.810200E-01, 3.344600E-01, 5.656400E-01, 1.172140E+00};
double chamber_abs_xs[7] = {5.113200E-04, 7.581300E-05, 3.164300E-04, 1.167500E-03, 3.397700E-03, 9.188600E-03, 2.324400E-02};
double chamber_scatter_xs[49] = {6.616590E-02, 5.907000E-02, 2.833400E-04, 1.462200E-06, 2.064200E-08, 0., 0.,
                                 0., 2.403770E-01, 5.243500E-02, 2.499000E-04, 1.923900E-05, 2.987500E-06, 4.214000E-07,  
                                 0., 0., 1.834250E-01, 9.228800E-02, 6.936500E-03, 1.079000E-03, 2.054300E-04,
                                 0., 0., 0., 7.907690E-02, 1.699900E-01, 2.586000E-02, 4.925600E-03,
                                 0., 0., 0., 3.734000E-05, 9.975700E-02, 2.067900E-01, 2.447800E-02,
                                 0., 0., 0., 0., 9.174200E-04, 3.167740E-01, 2.387600E-01,
                                 0., 0., 0., 0., 0., 4.979300E-02, 1.09910E+00};
double chamber_fis_xs[7] = {4.79002E-09, 5.82564E-09, 4.63719E-07, 5.24406E-06, 1.45390E-07, 7.14972E-07, 2.08041E-06};
double chamber_nufis_xs[7] = {1.323401E-08, 1.434500E-08, 1.128599E-06, 1.276299E-05, 3.538502E-07, 1.740099E-06, 5.063302E-06};
double chamber_chi[7] = {5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0., 0., 0.};

//###############################################################################
//##############################      Guide Tube      ###########################
//###############################################################################
double guide_total_xs[7] = {1.260320E-01, 2.931600E-01, 2.842500E-01, 2.810200E-01, 3.344600E-01, 5.656400E-01, 1.172140E+00};
double guide_abs_xs[7] = {5.113200E-04, 7.581300E-05, 3.164300E-04, 1.167500E-03, 3.397700E-03, 9.188600E-03, 2.324400E-02};
double guide_scatter_xs[49] = {6.616590E-02, 5.907000E-02, 2.833400E-04, 1.462200E-06, 2.064200E-08, 0., 0.,
                               0., 2.403770E-01, 5.243500E-02, 2.499000E-04, 1.923900E-05, 2.987500E-06, 4.214000E-07,  
                               0., 0., 1.834250E-01, 9.228800E-02, 6.936500E-03, 1.079000E-03, 2.054300E-04,
                               0., 0., 0., 7.907690E-02, 1.699900E-01, 2.586000E-02, 4.925600E-03,
                               0., 0., 0., 3.734000E-05, 9.975700E-02, 2.067900E-01, 2.447800E-02,
                               0., 0., 0., 0., 9.174200E-04, 3.167740E-01, 2.387600E-01,
                               0., 0., 0., 0., 0., 4.979300E-02, 1.09910E+00};
double guide_fis_xs[7] = {4.79002E-09, 5.82564E-09, 4.63719E-07, 5.24406E-06, 1.45390E-07, 7.14972E-07, 2.08041E-06};
double guide_nufis_xs[7] = {1.323401E-08, 1.434500E-08, 1.128599E-06, 1.276299E-05, 3.538502E-07, 1.740099E-06, 5.063302E-06};
double guide_chi[7] = {5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0., 0., 0.};

//###############################################################################
//################################      Water      ##############################
//###############################################################################
double water_total_xs[7] = {1.592060E-01, 4.129700E-01, 5.903100E-01, 5.843500E-01, 7.180000E-01, 1.254450E+00, 2.650380E+00};
double water_abs_xs[7] = {6.010500E-04, 1.579300E-05, 3.371600E-04, 1.940600E-03, 5.741600E-03, 1.500100E-02, 3.723900E-02};
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
    Circle circle1(0.0, 0.0, 0.4, 1);
    Circle circle2(0.0, 0.0, 0.3, 2);
    Circle circle3(0.0, 0.0, 0.2, 3);
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
    Universe u2(2);
    Universe u3(3);
    Cell *cell0 = new CellBasic(1, 1);
    Cell *cell1 = new CellBasic(1, 2);
    Cell *cell2 = new CellBasic(2, 1);
    Cell *cell3 = new CellBasic(2, 2);
    Cell *cell4 = new CellBasic(3, 1, 0, 8);
    Cell *cell5 = new CellBasic(3, 2);
    Cell *cell6 = new CellFill(0, 5);
    cell0->addSurface(-1, &circle1);
    cell1->addSurface(1, &circle1);
    cell2->addSurface(-1, &circle2);
    cell3->addSurface(1, &circle2);
    cell4->addSurface(-1, &circle3);
    cell5->addSurface(1, &circle3);
    cell6->addSurface(1, &left);
    cell6->addSurface(-1, &right);
    cell6->addSurface(1, &bottom);
    cell6->addSurface(-1, &top);
    std::cout << "Create lattices ..." << std::endl;
    Lattice lattice(5, 1.0, 1.0);
    int universe_id[16] = {1, 2, 1, 2, 2, 3, 2, 3, 1, 2, 1, 2, 2, 3, 2, 3};
    lattice.setLatticeCells(4, 4, universe_id);
    std::cout << "Create the coarse mesh ..." << std::endl;
    Mesh mesh(MOC, true);
    std::cout << "Create the geometry ..." << std::endl;
    Geometry geom(&mesh);
    geom.addMaterial(&uo2);
    geom.addMaterial(&water);
    geom.addCell(cell0);
    geom.addCell(cell1);
    geom.addCell(cell2);
    geom.addCell(cell3);
    geom.addCell(cell4);
    geom.addCell(cell5);
    geom.addCell(cell6);
    geom.addLattice(&lattice);
    geom.initializeFlatSourceRegions();
    std::cout << "Create the CMFD acceleration ..." << std::endl;
    Cmfd cmfd(&geom);
    cmfd.setOmega(1.0);
    std::cout << "Create the track generator ..." << std::endl;
    TrackGenerator tracks(&geom, num_azim, track_spacing);
    tracks.generateTracks();

    std::cout << "start to run the simulation ..." << std::endl;
    CPUSolver moc_solver(&geom, &tracks, &cmfd);
    moc_solver.setNumThreads(num_threads);
    moc_solver.setSourceConvergenceThreshold(tolerance);
    moc_solver.convergeSource(max_iters);
    moc_solver.printTimerReport();
}

void pin_cell_demo()
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
    Circle circle(0.0, 0.0, 1.0);
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
    Lattice lattice(2, 4.0, 4.0);
    int universe_id = 1;
    lattice.setLatticeCells(1, 1, &universe_id);
    std::cout << "Create the geometry ..." << std::endl;
    Geometry geom;
    // Add materials
    geom.addMaterial(&uo2);
    geom.addMaterial(&water);
    // Add cells
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

void nested_lattice_demo()
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
    Circle circle1(0.0, 0.0, 0.4, 1);
    Circle circle2(0.0, 0.0, 0.3, 2);
    Circle circle3(0.0, 0.0, 0.2, 3);
    XPlane left(-2.0);
    XPlane right(2.0);
    YPlane bottom(-2.0);
    YPlane top(2.0);
    left.setBoundaryType(REFLECTIVE);
    right.setBoundaryType(REFLECTIVE);
    top.setBoundaryType(REFLECTIVE);
    bottom.setBoundaryType(REFLECTIVE);
    std::cout << "Create cells ..." << std::endl;
    Universe u0(0);
    Universe u1(1);
    Universe u2(2);
    Universe u3(3);
    Cell *cell0 = new CellBasic(1, 1);
    Cell *cell1 = new CellBasic(1, 2);
    Cell *cell2 = new CellBasic(2, 1);
    Cell *cell3 = new CellBasic(2, 2);
    Cell *cell4 = new CellBasic(3, 1);
    Cell *cell5 = new CellBasic(3, 2);
    Cell *cell6 = new CellFill(5, 4);
    Cell *cell7 = new CellFill(0, 6);
    cell0->addSurface(-1, &circle1);
    cell1->addSurface(1, &circle1);
    cell2->addSurface(-1, &circle2);
    cell3->addSurface(1, &circle2);
    cell4->addSurface(-1, &circle3);
    cell5->addSurface(1, &circle3);

    cell7->addSurface(1, &left);
    cell7->addSurface(-1, &right);
    cell7->addSurface(1, &bottom);
    cell7->addSurface(-1, &top);
    std::cout << "Create lattices ..." << std::endl;
    // 2x2 assembly
    Lattice assembly(4, 1.0, 1.0);
    int universe_id1[4] = {1, 2, 1, 3};
    assembly.setLatticeCells(2, 2, universe_id1);
    // 2x2 core
    Lattice core(6, 2.0, 2.0);
    int universe_id2[4] = {5, 5, 5, 5};
    core.setLatticeCells(2, 2, universe_id2);

    std::cout << "Create the geometry ..." << std::endl;
    Geometry geom;
    geom.addMaterial(&uo2);
    geom.addMaterial(&water);
    geom.addCell(cell0);
    geom.addCell(cell1);
    geom.addCell(cell2);
    geom.addCell(cell3);
    geom.addCell(cell4);
    geom.addCell(cell5);
    geom.addCell(cell6);
    geom.addCell(cell7);
    geom.addLattice(&assembly);
    geom.addLattice(&core);
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

void large_lattice_demo()
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
    Circle circle1(0.0, 0.0, 0.4, 1);
    Circle circle2(0.0, 0.0, 0.3, 2);
    Circle circle3(0.0, 0.0, 0.2, 3);
    XPlane left(-2.0);
    XPlane right(2.0);
    YPlane bottom(-2.0);
    YPlane top(2.0);
    left.setBoundaryType(REFLECTIVE);
    right.setBoundaryType(REFLECTIVE);
    top.setBoundaryType(REFLECTIVE);
    bottom.setBoundaryType(REFLECTIVE);
    std::cout << "Create cells ..." << std::endl;
    Universe u0(0);
    Universe u1(1);
    Universe u2(2);
    Universe u3(3);
    Universe u5(5);
    Cell *cell0 = new CellBasic(1, 1);
    Cell *cell1 = new CellBasic(1, 2);
    Cell *cell2 = new CellBasic(2, 1);
    Cell *cell3 = new CellBasic(2, 2);
    Cell *cell4 = new CellBasic(3, 1);
    Cell *cell5 = new CellBasic(3, 2);
    Cell *cell6 = new CellFill(5, 4);
    Cell *cell7 = new CellFill(0, 6);
    cell0->addSurface(-1, &circle1);
    cell1->addSurface(1, &circle1);
    cell2->addSurface(-1, &circle2);
    cell3->addSurface(1, &circle2);
    cell4->addSurface(-1, &circle3);
    cell5->addSurface(1, &circle3);

    cell7->addSurface(1, &left);
    cell7->addSurface(-1, &right);
    cell7->addSurface(1, &bottom);
    cell7->addSurface(-1, &top);
    std::cout << "Create lattices ..." << std::endl;
    // 2x2 assembly
    Lattice assembly(4, 1.0, 1.0);
    int universe_id1[4] = {1, 2, 1, 3};
    assembly.setLatticeCells(2, 2, universe_id1);
    // 2x2 core
    Lattice core(6, 2.0, 2.0);
    int universe_id2[4] = {5, 5, 5, 5};
    core.setLatticeCells(2, 2, universe_id2);

    std::cout << "Create the geometry ..." << std::endl;
    Geometry geom;
    geom.addMaterial(&uo2);
    geom.addMaterial(&water);
    geom.addCell(cell0);
    geom.addCell(cell1);
    geom.addCell(cell2);
    geom.addCell(cell3);
    geom.addCell(cell4);
    geom.addCell(cell5);
    geom.addCell(cell6);
    geom.addCell(cell7);
    geom.addLattice(&assembly);
    geom.addLattice(&core);
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

void full_core_demo()
{
    std::cout << "Set the main simulation parameters ..." << std::endl;
    int num_threads = 1;
    double track_spacing = 0.1;
    int num_azim = 4;
    double tolerance = 1.0E-5;
    int max_iters = 200;
    set_log_level("NORMAL_LOG");
    std::cout << "Simulating a mock full core PWR ..." << std::endl;
    std::cout << "Create the materials ..." << std::endl;
    Material uo2(1);
    Material mox43(2);
    Material mox7(3);
    Material mox87(4);
    Material gtube(5);
    Material fchamber(6);
    Material water(7);
    uo2.setNumEnergyGroups(7);
    uo2.setSigmaT(uo2_total_xs, 7);
    uo2.setSigmaA(uo2_abs_xs, 7);
    uo2.setSigmaS(uo2_scatter_xs, 49);
    uo2.setSigmaF(uo2_fis_xs, 7);
    uo2.setNuSigmaF(uo2_nufis_xs, 7);
    uo2.setChi(uo2_chi, 7);
    mox43.setNumEnergyGroups(7);
    mox43.setSigmaT(mox43_total_xs, 7);
    mox43.setSigmaA(mox43_abs_xs, 7);
    mox43.setSigmaS(mox43_scatter_xs, 49);
    mox43.setSigmaF(mox43_fis_xs, 7);
    mox43.setNuSigmaF(mox43_nufis_xs, 7);
    mox43.setChi(mox43_chi, 7);
    mox7.setNumEnergyGroups(7);
    mox7.setSigmaT(mox7_total_xs, 7);
    mox7.setSigmaA(mox7_abs_xs, 7);
    mox7.setSigmaS(mox7_scatter_xs, 49);
    mox7.setSigmaF(mox7_fis_xs, 7);
    mox7.setNuSigmaF(mox7_nufis_xs, 7);
    mox7.setChi(mox7_chi, 7);
    mox87.setNumEnergyGroups(7);
    mox87.setSigmaT(mox87_total_xs, 7);
    mox87.setSigmaA(mox87_abs_xs, 7);
    mox87.setSigmaS(mox87_scatter_xs, 49);
    mox87.setSigmaF(mox87_fis_xs, 7);
    mox87.setNuSigmaF(mox87_nufis_xs, 7);
    mox87.setChi(mox87_chi, 7);
    gtube.setNumEnergyGroups(7);
    gtube.setSigmaT(guide_total_xs, 7);
    gtube.setSigmaA(guide_abs_xs, 7);
    gtube.setSigmaS(guide_scatter_xs, 49);
    gtube.setSigmaF(guide_fis_xs, 7);
    gtube.setNuSigmaF(guide_nufis_xs, 7);
    gtube.setChi(guide_chi, 7);
    fchamber.setNumEnergyGroups(7);
    fchamber.setSigmaT(chamber_total_xs, 7);
    fchamber.setSigmaA(chamber_abs_xs, 7);
    fchamber.setSigmaS(chamber_scatter_xs, 49);
    fchamber.setSigmaF(chamber_fis_xs, 7);
    fchamber.setNuSigmaF(chamber_nufis_xs, 7);
    fchamber.setChi(chamber_chi, 7);
    water.setNumEnergyGroups(7);
    water.setSigmaT(water_total_xs, 7);
    water.setSigmaA(water_abs_xs, 7);
    water.setSigmaS(water_scatter_xs, 49);
    water.setSigmaF(water_fis_xs, 7);
    water.setNuSigmaF(water_nufis_xs, 7);
    water.setChi(water_chi, 7);
    std::cout << "Create surfaces ..." << std::endl;
    Circle circle1(0.0, 0.0, 0.54, 1);
    Circle circle2(0.0, 0.0, 0.58, 2);
    Circle circle3(0.0, 0.0, 0.62, 3);
    XPlane left(-224.91);
    XPlane right(224.91);
    YPlane bottom(-224.91);
    YPlane top(224.91);
    left.setBoundaryType(VACUUM);
    right.setBoundaryType(VACUUM);
    top.setBoundaryType(VACUUM);
    bottom.setBoundaryType(VACUUM);
    std::cout << "Create cells ..." << std::endl;
    // UO2 pin cells
    Universe u0(0);
    Universe u1(1);
    Universe u2(2);
    Universe u3(3);
    Universe u4(4);
    Universe u5(5);
    Universe u6(6);
    Universe u7(7);
    Universe u10(10);
    Universe u11(11);
    Universe u12(12);
    Universe u13(13);
    Universe u14(14);
    
    Cell *cell0 = new CellBasic(1, 1, 3, 8);
    Cell *cell1 = new CellBasic(1, 7, 0, 8);
    Cell *cell2 = new CellBasic(1, 7, 0, 8);
    Cell *cell3 = new CellBasic(1, 7, 0, 8);
    cell0->addSurface(-1, &circle1);
    cell1->addSurface(1, &circle1);
    cell1->addSurface(-1, &circle2);
    cell2->addSurface(1, &circle2);
    cell2->addSurface(-1, &circle3);
    cell3->addSurface(1, &circle3);
    // 4.3% MOX pin cells
    Cell *cell4 = new CellBasic(2, 2, 3, 8);
    Cell *cell5 = new CellBasic(2, 7, 0, 8);
    Cell *cell6 = new CellBasic(2, 7, 0, 8);
    Cell *cell7 = new CellBasic(2, 7, 0, 8);
    cell4->addSurface(-1, &circle1);
    cell5->addSurface(1, &circle1);
    cell5->addSurface(-1, &circle2);
    cell6->addSurface(1, &circle2);
    cell6->addSurface(-1, &circle3);
    cell7->addSurface(1, &circle3);
    // 7% MOX pin cells
    Cell *cell8 = new CellBasic(3, 3, 3, 8);
    Cell *cell9 = new CellBasic(3, 7, 0, 8);
    Cell *cell10 = new CellBasic(3, 7, 0, 8);
    Cell *cell11 = new CellBasic(3, 7, 0, 8);
    cell8->addSurface(-1, &circle1);
    cell9->addSurface(1, &circle1);
    cell9->addSurface(-1, &circle2);
    cell10->addSurface(1, &circle2);
    cell10->addSurface(-1, &circle3);
    cell11->addSurface(1, &circle3);
    // 8.7% MOX pin cells
    Cell *cell12 = new CellBasic(4, 4, 3, 8);
    Cell *cell13 = new CellBasic(4, 7, 0, 8);
    Cell *cell14 = new CellBasic(4, 7, 0, 8);
    Cell *cell15 = new CellBasic(4, 7, 0, 8);
    cell12->addSurface(-1, &circle1);
    cell13->addSurface(1, &circle1);
    cell13->addSurface(-1, &circle2);
    cell14->addSurface(1, &circle2);
    cell14->addSurface(-1, &circle3);
    cell15->addSurface(1, &circle3);
    // Fission chamber pin cells
    Cell *cell16 = new CellBasic(5, 6, 3, 8);
    Cell *cell17 = new CellBasic(5, 7, 0, 8);
    Cell *cell18 = new CellBasic(5, 7, 0, 8);
    Cell *cell19 = new CellBasic(5, 7, 0, 8);
    cell16->addSurface(-1, &circle1);
    cell17->addSurface(1, &circle1);
    cell17->addSurface(-1, &circle2);
    cell18->addSurface(1, &circle2);
    cell18->addSurface(-1, &circle3);
    cell19->addSurface(1, &circle3);
    // Guide tube pin cells
    Cell *cell20 = new CellBasic(6, 5, 3, 8);
    Cell *cell21 = new CellBasic(6, 7, 0, 8);
    Cell *cell22 = new CellBasic(6, 7, 0, 8);
    Cell *cell23 = new CellBasic(6, 7, 0, 8);
    cell20->addSurface(-1, &circle1);
    cell21->addSurface(1, &circle1);
    cell21->addSurface(-1, &circle2);
    cell22->addSurface(1, &circle2);
    cell22->addSurface(-1, &circle3);
    cell23->addSurface(1, &circle3);
    // Moderator cell
    Cell *cell24 = new CellBasic(7, 7);
    // Lattice is related with cells
    Cell *cell25 = new CellFill(10, 20);
    Cell *cell26 = new CellFill(11, 21);
    Cell *cell27 = new CellFill(12, 22);
    Cell *cell28 = new CellFill(13, 23);
    Cell *cell29 = new CellFill(14, 24);
    // Full geometry
    Cell *cell30 = new CellFill(0, 30);
    cell30->addSurface(1, &left);
    cell30->addSurface(-1, &right);
    cell30->addSurface(1, &bottom);
    cell30->addSurface(-1, &top);
    std::cout << "Create lattices ..." << std::endl;
    Lattice lattice1(20, 1.26, 1.26);
    int universe_id1[17*17] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1, 1, 1,
                               1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 6, 1, 1, 6, 1, 1, 5, 1, 1, 6, 1, 1, 6, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1,
                               1, 1, 1, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    lattice1.setLatticeCells(17, 17, universe_id1);
    Lattice lattice2(21, 1.26, 1.26);
    int universe_id2[17*17] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
                               2, 3, 3, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 3, 3, 2,
                               2, 3, 3, 6, 3, 4, 4, 4, 4, 4, 4, 4, 3, 6, 3, 3, 2,
                               2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2,
                               2, 3, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 3, 2,
                               2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2,
                               2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2,
                               2, 3, 6, 4, 4, 6, 4, 4, 5, 4, 4, 6, 4, 4, 6, 3, 2,
                               2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2,
                               2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2,
                               2, 3, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 3, 2,
                               2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2,
                               2, 3, 3, 6, 3, 4, 4, 4, 4, 4, 4, 4, 3, 6, 3, 3, 2,
                               2, 3, 3, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 3, 3, 2,
                               2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    lattice2.setLatticeCells(17, 17, universe_id2);
    Lattice lattice3(22, 0.252, 0.252);
    int universe_id3[25] = {7, 7, 7, 7, 7,
                            7, 7, 7, 7, 7,
                            7, 7, 7, 7, 7,
                            7, 7, 7, 7, 7,
                            7, 7, 7, 7, 7};
    lattice3.setLatticeCells(5, 5, universe_id3);
    Lattice lattice4(23, 1.26, 1.26);
    int universe_id4[17*17] = {12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
    lattice4.setLatticeCells(17, 17, universe_id4);
    Lattice lattice5(24, 1.26, 1.26);
    int universe_id5[17*17] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    lattice5.setLatticeCells(17, 17, universe_id5);
    // 21x21 core to represent two bundles and water
    Lattice lattice6(30, 21.42, 21.42);
    int universe_id6[21*21] = {13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                               13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13,
                               13, 13, 13, 13, 14, 14, 14, 10, 11, 10, 11, 10, 11, 10, 14, 14, 13, 13, 13, 13, 13,
                               13, 13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13, 13, 13,
                               13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 14, 13, 13,
                               13, 13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13,
                               13, 14, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13, 13,
                               13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13,
                               13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13,
                               13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13,
                               13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13,
                               13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13,
                               13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13,
                               13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13,
                               13, 14, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13, 13,
                               13, 13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13,
                               13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13, 13,
                               13, 13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13, 13, 13,
                               13, 13, 13, 13, 14, 14, 14, 10, 11, 10, 11, 10, 11, 10, 14, 14, 13, 13, 13, 13, 13,
                               13, 13, 14, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13,
                               13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13};
    lattice6.setLatticeCells(21, 21, universe_id6);
    std::cout << "Create the geometry ..." << std::endl;
    Geometry geom;
    geom.addMaterial(&uo2);
    geom.addMaterial(&mox43);
    geom.addMaterial(&mox7);
    geom.addMaterial(&mox87);
    geom.addMaterial(&gtube);
    geom.addMaterial(&fchamber);
    geom.addMaterial(&water);
    geom.addCell(cell0); geom.addCell(cell1); geom.addCell(cell2);
    geom.addCell(cell3); geom.addCell(cell4); geom.addCell(cell5);
    geom.addCell(cell6); geom.addCell(cell7); geom.addCell(cell8);
    geom.addCell(cell9); geom.addCell(cell10); geom.addCell(cell11);
    geom.addCell(cell12); geom.addCell(cell13); geom.addCell(cell14);
    geom.addCell(cell15); geom.addCell(cell16); geom.addCell(cell17);
    geom.addCell(cell18); geom.addCell(cell19); geom.addCell(cell20);
    geom.addCell(cell21); geom.addCell(cell22); geom.addCell(cell23);
    geom.addCell(cell24); geom.addCell(cell25); geom.addCell(cell26);
    geom.addCell(cell27); geom.addCell(cell28); geom.addCell(cell29);
    geom.addCell(cell30);
    geom.addLattice(&lattice1);
    geom.addLattice(&lattice2);
    geom.addLattice(&lattice3);
    geom.addLattice(&lattice4);
    geom.addLattice(&lattice5);
    geom.addLattice(&lattice6);
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

void bundled_lattice_demo()
{
    std::cout << "Set the main simulation parameters ..." << std::endl;
    int num_threads = 1;
    double track_spacing = 0.1;
    int num_azim = 4;
    double tolerance = 1.0E-5;
    int max_iters = 200;
    set_log_level("NORMAL_LOG");
    std::cout << "Simulating a mock PWR lattice structure ..." << std::endl;
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
    Circle circle1(0.0, 0.0, 0.15, 1);
    Circle circle2(0.0, 0.0, 0.2, 2);
    Circle circle3(0.0, 0.0, 0.25, 3);
    Circle circle4(0.0, 0.0, 0.3, 4);
    Circle circle5(0.0, 0.0, 0.35, 5);
    Circle circle6(0.0, 0.0, 0.4, 6);
    XPlane left(-34.0);
    XPlane right(34.0);
    YPlane bottom(-34.0);
    YPlane top(34.0);
    left.setBoundaryType(REFLECTIVE);
    right.setBoundaryType(REFLECTIVE);
    top.setBoundaryType(REFLECTIVE);
    bottom.setBoundaryType(REFLECTIVE);
    std::cout << "Create cells ..." << std::endl;
    // UO2 pin cells
    Universe u0(0);
    Universe u1(1);
    Universe u2(2);
    Universe u3(3);
    Universe u4(4);
    Universe u5(5);
    Universe u6(6);
    Universe u7(7);
    Universe u10(10);
    Universe u11(11);
    
    Cell *cell0 = new CellBasic(1, 1);
    Cell *cell1 = new CellBasic(1, 2);
    Cell *cell2 = new CellBasic(2, 1);
    Cell *cell3 = new CellBasic(2, 2);
    Cell *cell4 = new CellBasic(3, 1);
    Cell *cell5 = new CellBasic(3, 2);
    Cell *cell6 = new CellBasic(4, 1);
    Cell *cell7 = new CellBasic(4, 2);
    Cell *cell8 = new CellBasic(5, 1);
    Cell *cell9 = new CellBasic(5, 2);
    Cell *cell10 = new CellBasic(6, 1);
    Cell *cell11 = new CellBasic(6, 2);
    Cell *cell12 = new CellFill(9, 7);
    Cell *cell13 = new CellFill(11, 8);
    Cell *cell14 = new CellFill(0, 10);

    cell0->addSurface(-1, &circle1);
    cell1->addSurface(1, &circle1);
    cell2->addSurface(-1, &circle2);
    cell3->addSurface(1, &circle2);
    cell4->addSurface(-1, &circle3);
    cell5->addSurface(1, &circle3);
    cell6->addSurface(-1, &circle4);
    cell7->addSurface(1, &circle4);
    cell8->addSurface(-1, &circle5);
    cell9->addSurface(1, &circle5);
    cell10->addSurface(-1, &circle6);
    cell11->addSurface(1, &circle6);
    cell14->addSurface(1, &left);
    cell14->addSurface(-1, &right);
    cell14->addSurface(1, &bottom);
    cell14->addSurface(-1, &top);
    std::cout << "Create lattices ..." << std::endl;
    Lattice assembly1(7, 1.0, 1.0);
    int universe_id1[17*17] = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                               2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                               1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
    assembly1.setLatticeCells(17, 17, universe_id1);

    Lattice assembly2(8, 1.0, 1.0);
    int universe_id2[17*17] = {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
                               5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5,
                               4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4};
    assembly2.setLatticeCells(17, 17, universe_id2);
    // 4x4 core
    Lattice core(10, 17.0, 17.0);
    int universe_id3[21*21] = {9, 11, 9, 11,
                               11, 9, 11, 9,
                               9, 11, 9, 11,
                               11, 9, 11, 9};
    core.setLatticeCells(4, 4, universe_id3);
    std::cout << "Create the geometry ..." << std::endl;
    Geometry geom;
    geom.addMaterial(&uo2);
    geom.addMaterial(&water);
    geom.addCell(cell0); geom.addCell(cell1); geom.addCell(cell2);
    geom.addCell(cell3); geom.addCell(cell4); geom.addCell(cell5);
    geom.addCell(cell6); geom.addCell(cell7); geom.addCell(cell8);
    geom.addCell(cell9); geom.addCell(cell10); geom.addCell(cell11);
    geom.addCell(cell12); geom.addCell(cell13); geom.addCell(cell14);
    geom.addLattice(&assembly1);
    geom.addLattice(&assembly2);
    geom.addLattice(&core);
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

int main(int argc, char *argv[])
{
    std::cout << "Run openmoc 2D neutron transport simulation !" << std::endl;

    tiny_lattice_demo();
    simple_lattice_demo();
    pin_cell_demo();
    nested_lattice_demo();
    large_lattice_demo();
    // Because too much time costing, so just to comment it when normal runs, Pingzhou Ming, 2021.7.8
    // full_core_demo();
    bundled_lattice_demo();

    std::cout << "Finished !" << std::endl;

    return 0;
}
