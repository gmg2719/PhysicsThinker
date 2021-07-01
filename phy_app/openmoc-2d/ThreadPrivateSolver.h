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

#ifndef _THREADPRIVATESOLVER_H_
#define _THREADPRIVATESOLVER_H_     1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file ThreadPrivateSolver.h
 * @brief The ThreadPrivateSolver class.
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

/** Indexing scheme for the thread private FSR scalar flux for each thread */
#define _thread_flux(tid,r,e) (_thread_flux[(tid)][(r)*_num_groups+(e)])

/** Indexing scheme for the thread private Cmfd Mesh surface currents for each 
 * thread in each FSR and energy group */
#define _thread_currents(tid,r,e) (_thread_currents[(tid)*_num_mesh_cells*8*_cmfd->getNumCmfdGroups() + (r)*_cmfd->getNumCmfdGroups() + _cmfd->getCmfdGroup((e))])

/**
 * @class ThreadPrivateSolver ThreadPrivateSolver.h "openmoc/src/ThreadPrivateSolver.h"
 * @brief This is a subclass of the CPUSolver which uses thread private
 *        arrays for the FSR scalar fluxes to minimize OpenMPC atomics.
 * @details Since this class stores a separate copy of the FSR scalar
 *          fluxes for each OMP thread, the memory requirements are greater
 *          than for the CPUSolver, but the parallel performance and scaling
 *          are much better.
 */
class ThreadPrivateSolver : public CPUSolver {

protected:

  /** An array for the FSR scalar fluxes for each thread */
  FP_PRECISION** _thread_flux;

  /** An array for the CMFD Mesh surface currents for each thread */
  FP_PRECISION* _thread_currents;

  void initializeFluxArrays();
  void initializeCmfd();

  void flattenFSRFluxes(FP_PRECISION value);
  void zeroSurfaceCurrents();
  void scalarFluxTally(segment* curr_segment, int azim_index,
                       FP_PRECISION* track_flux,
                       FP_PRECISION* fsr_flux, bool fwd);
  void reduceThreadScalarFluxes();
  void reduceThreadSurfaceCurrents();
  void transportSweep();

public:
  ThreadPrivateSolver(Geometry* geometry=NULL,
                      TrackGenerator* track_generator=NULL,
                      Cmfd* cmfd=NULL);
  virtual ~ThreadPrivateSolver();
};


#endif /* THREADPRIVATESOLVER_H_ */
