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

#ifndef _VECTORIZEDPRIVATESOLVER_H_
#define _VECTORIZEDPRIVATESOLVER_H_     1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file VectorizedPrivateSolver.h
 * @brief The VectorizedPrivateSolver class.
 * @date July 18, 2013
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifdef __cplusplus
  #include "VectorizedSolver.h"
#endif


/** Indexing scheme for the thread private scalar flux for each thread in
 *  each flat source region and in each energy group */
#define _thread_flux(tid,r,e) (_thread_flux[(tid)][(r)*_num_groups+(e)])

/**
 * @class VectorizedPrivateSolver VectorizedPrivateSolver.h "src/VectorizedPrivateSolver.h"
 * @brief This is a subclass of the VectorizedSolver class. This class uses a
 *        thread private array for FSR scalar fluxes during each transport sweep
 *        to avoid the use of OpenMP atomics, akin to the ThreadPrivateSolver
 *        class. It also uses memory-aligned data structures and Intel's
 *        auto-vectorization
 */
class VectorizedPrivateSolver : public VectorizedSolver {

private:

  /** An array for the FSR scalar fluxes for each thread */
  FP_PRECISION** _thread_flux;

  void initializeFluxArrays();

  void flattenFSRFluxes(FP_PRECISION value);

  void scalarFluxTally(segment* curr_segment, int azim_index,
                      FP_PRECISION* track_flux, FP_PRECISION* fsr_flux);

  void transportSweep();
  void reduceThreadScalarFluxes();


public:
  VectorizedPrivateSolver(Geometry* geometry=NULL,
                          TrackGenerator* track_generator=NULL,
                          Cmfd* cmfd=NULL);
  virtual ~VectorizedPrivateSolver();
};


#endif /* VECTORIZEDPRIVATESOLVER_H_ */
