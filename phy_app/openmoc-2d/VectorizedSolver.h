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

#ifndef _VECTORIZEDSOLVER_H_
#define _VECTORIZEDSOLVER_H_      1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file VectorizedSolver.h
 * @brief The VectorizedSolver class.
 * @date May 28, 2013
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifdef __cplusplus
  #include <cmath>
#ifdef _OPENMP
  #include <omp.h>
#endif
  #include <cstdlib>
  #include "CPUSolver.h"
#endif

/** Indexing scheme for the optical length (\f$ l\Sigma_t \f$) for a
 *  given Track segment for each polar angle and energy group */
#define taus(p,e) (taus[(p)*_num_groups + (e)])

/** Indexing scheme for the exponentials in the neutron transport equation
 *  (\f$ 1 - exp(-\frac{l\Sigma_t}{sin(\theta_p)}) \f$) for a given
 *  Track segment for each polar angle and energy group */
#define exponentials(p,e) (exponentials[(p)*_num_groups + (e)])

/**
 * @class VectorizedSolver VectorizedSolver.h "src/VectorizedSolver.h"
 * @brief This is a subclass of the CPUSolver class which uses memory-aligned
 *        data structures and Intel's auto-vectorization.
 * @note This class is only compiled if the Intel compiler is used when building
 *       OpenMOC. If building OpenMOC with the "--with-icpc" flag, then this
 *       class will be available in the "openmoc.intel.single" or
 *       "openmoc.intel.double" Python module.
 */
class VectorizedSolver : public CPUSolver {

protected:

  /** Number of energy groups divided by vector widths (VEC_LENGTH) */
  int _num_vector_lengths;

  /** An array for the optical length for each thread in each energy group */
  FP_PRECISION* _thread_taus;

  /** An array for the exponential terms in the transport equation for *
   *  each thread in each energy group and polar angle */
  FP_PRECISION* _thread_exponentials;

  void buildExpInterpTable();
  void initializeFluxArrays();
  void initializeSourceArrays();

  void normalizeFluxes();
  FP_PRECISION computeFSRSources();
  void scalarFluxTally(segment* curr_segment, int azim_index,
                       FP_PRECISION* track_flux,
                       FP_PRECISION* fsr_flux, bool fwd);
  void transferBoundaryFlux(int track_id, int azim_index, bool direction,
                            FP_PRECISION* track_flux);
  void addSourceToScalarFlux();
  void computeKeff();


  /**
   * @brief Computes an array of the exponentials in the transport equation,
   *        \f$ exp(-\frac{\Sigma_t * l}{sin(\theta)}) \f$, for each
   *        energy group and polar angle for a given segment.
   * @param curr_segment pointer to the segment of interest
   * @param exponentials the array to store the exponential values
   */
  virtual void computeExponentials(segment* curr_segment,
                                   FP_PRECISION* exponentials);

public:
  VectorizedSolver(Geometry* geometry=NULL,
                   TrackGenerator* track_generator=NULL,
                   Cmfd* cmfd=NULL);

  virtual ~VectorizedSolver();

  int getNumVectorWidths();

  void setGeometry(Geometry* geometry);
};


#endif /* VECTORIZEDSOLVER_H_ */
