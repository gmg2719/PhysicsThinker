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

#ifndef _CMFD_H_
#define _CMFD_H_      1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//
/**
 * @file Cmfd.h
 * @brief The Cmfd class.
 * @date October 14, 2013
 * @author Sam Shaner, MIT, Course 22 (shaner@mit.edu)
 */

#ifdef __cplusplus
  #include <utility>
  #include <cmath>
  #include <climits>
  #include <string>
  #include <sstream>
  #include <queue>
  #include <iostream>
  #include <fstream>
  #include "na/Quadrature.h"
  #include "common/log.h"
  #include "common/my_timer.h"
  #include "geometry/Mesh.h"
  #include "geometry/Material.h"
  #include "geometry/Surface.h"
  #include "geometry/Geometry.h"
#endif


/**
 * @enum eigenMethod
 * @brief Eigenvalue solution methods.
*/
enum eigenMethod {

  /** The power iteration method */
  POWER,

  /** The Wielandt iteration method */
  WIELANDT
};



/**
 * @class Cmfd Cmfd.h "src/Cmfd.h"
 * @brief A class for Coarse Mesh Finite Difference (CMFD) acceleration.
 */
class Cmfd {

private:

  /** Pointer to polar Quadrature object */
  Quadrature* _quad;

  /** Pointer to Geometry object */
  Geometry* _geometry;

  /** Pointer to Mesh object */
  Mesh* _mesh;

  /** The keff eigenvalue */
  double _k_eff;

  /** The A matrix */
  double** _A;

  /** The M matrix */
  double** _M;

  /** The AM matrix */
  double** _AM;

  /** The old source vector */
  double* _old_source;

  /** The new source vector */
  double* _new_source;

  /** A temporary scalar flux vector */
  double* _phi_temp;

  /** Gauss-Seidel SOR factor */
  double _omega;

  /** L2 norm */
  double _l2_norm;

  /** Keff convergence criteria */
  double _conv_criteria;

  /** Number of cells in x direction */
  int _cx;

  /** Number of cells in y direction */
  int _cy;

  /** Pointer to Timer object */
  Timer* _timer;

  /** Flux type (PRIMAL or ADJOINT) */
  fluxType _flux_type;

  /** Number of energy groups */
  int _num_groups;

  /** Number of coarse energy groups */
  int _num_cmfd_groups;

  /** Coarse energy indices for fine energy groups */
  int* _group_indices;

  /** Map of MOC groups to CMFD groups */
  int* _group_indices_map;

  /** Number of FSRs */
  int _num_FSRs;

  /** Solution method (DIFFUSION or MOC) */
  solveType _solve_method;

  /** The volumes (areas) for each FSR */
  FP_PRECISION* _FSR_volumes;

  /** Pointers to Materials for each FSR */
  Material** _FSR_materials;

  /** The FSR scalar flux in each energy group */
  FP_PRECISION* _FSR_fluxes;

  /* Eigenvalue method */
  eigenMethod _eigen_method;

public:

  Cmfd(Geometry* geometry, double criteria=1e-8);
  virtual ~Cmfd();

  /* Worker functions */
  void constructMatrices();
  void computeDs();
  void computeXS();
  void updateMOCFlux();
  double computeDiffCorrect(double d, double h);
  double computeKeff();
  void initializeFSRs();
  void rescaleFlux();
  void linearSolve(double** mat, double* vec_x, double* vec_b,
                   double conv, int max_iter=10000);

  /* Matrix and Vector functions */
  void dumpVec(double* vec, int length);
  void matZero(double** mat, int width);
  void vecCopy(double* vec_from, double* vec_to);
  double vecSum(double* vec);
  void matMultM(double** mat, double* vec_x, double* vec_y);
  void matMultA(double** mat, double* vec_x, double* vec_y);
  void vecNormal(double** mat, double* vec);
  void vecSet(double* vec, double val);
  void vecScale(double* vec, double scale_val);
  void matSubtract(double** AM, double** A, double omega, double** M);
  double vecMax(double* vec);
  double rayleighQuotient(double* x, double* snew, double* sold);
  void createGroupStructure(int* group_indices, int ncg);
  
  /* Get parameters */
  Mesh* getMesh();
  double getKeff();
  int getNumCmfdGroups();
  int getCmfdGroup(int group);

  /* Set parameters */
  void setOmega(double omega);
  void setFluxType(const char* flux_type);
  void setEigenMethod(const char* eigen_method);

  /* Set FSR parameters */
  void setFSRMaterials(Material** FSR_materials);
  void setFSRVolumes(FP_PRECISION* FSR_volumes);
  void setFSRFluxes(FP_PRECISION* scalar_flux);
};

#endif /* CMFD_H_ */
