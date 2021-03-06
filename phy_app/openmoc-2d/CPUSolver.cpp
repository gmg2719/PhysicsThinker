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

#include "CPUSolver.h"


/**
 * @brief Constructor initializes array pointers for Tracks and Materials.
 * @details The constructor retrieves the number of energy groups and FSRs
 *          and azimuthal angles from the Geometry and TrackGenerator if
 *          passed in as parameters by the user. The constructor initalizes
 *          the number of OpenMP threads to a default of 1.
 * @param geometry an optional pointer to the Geometry
 * @param track_generator an optional pointer to the TrackGenerator
 * @param cmfd an optional pointer to a Cmfd object object
 */
CPUSolver::CPUSolver(Geometry* geometry, TrackGenerator* track_generator,
                     Cmfd* cmfd) : Solver(geometry, track_generator, cmfd) {

  setNumThreads(1);

#ifdef _OPENMP
  _FSR_locks = NULL;
  _mesh_surface_locks = NULL;
#endif
  _thread_fsr_flux = NULL;
}


/**
 * @brief Destructor deletes array for OpenMP mutual exclusion locks for
 *        FSR scalar flux updates, and calls Solver parent class destructor
 *        to deletes arrays for fluxes and sources.
 */
CPUSolver::~CPUSolver() {
#ifdef _OPENMP
  if (_FSR_locks != NULL)
    delete [] _FSR_locks;

  if (_mesh_surface_locks != NULL)
    delete [] _mesh_surface_locks;
  #endif

  if (_thread_fsr_flux != NULL)
    delete [] _thread_fsr_flux;

  if (_surface_currents != NULL)
    delete [] _surface_currents;
}


/**
 * @brief Returns the number of shared memory OpenMP threads in use.
 * @return the number of threads
 */
int CPUSolver::getNumThreads() {
  return _num_threads;
}


/**
 * @brief Returns the scalar flux for some FSR and energy group.
 * @param fsr_id the ID for the FSR of interest
 * @param energy_group the energy group of interest
 * @return the FSR scalar flux
 */
FP_PRECISION CPUSolver::getFSRScalarFlux(int fsr_id, int energy_group) {

  /* Error checking */
  if (fsr_id >= _num_FSRs)
    log_printf(ERROR_LOG, "Unable to return a scalar flux for FSR ID = %d in energy"
               " group %d since the solver only contains FSR with IDs less"
               " than or equal to %d", fsr_id, energy_group, _num_FSRs-1);

  if (fsr_id < 0)
    log_printf(ERROR_LOG, "Unable to return a scalar flux for FSR ID = %d "
               "in energy group %d since FSRs do not have ",
               "negative IDs", fsr_id, energy_group);

  if (energy_group-1 >= _num_groups)
    log_printf(ERROR_LOG, "Unable to return a scalar flux for FSR ID = %d "
               "in energy group %d since the solver only has %d energy "
               "groups", fsr_id, energy_group, _num_groups);

  if (energy_group <= 0)
    log_printf(ERROR_LOG, "Unable to return a scalar flux for FSR ID = %d "
               "in energy group %d since energy groups are greater than "
               "or equal to 1", fsr_id, energy_group);

  return _scalar_flux(fsr_id,energy_group-1);
}


/**
 * @brief Returns the source for some energy group for a flat source region
 * @param fsr_id the ID for the FSR of interest
 * @param energy_group the energy group of interest
 * @return the flat source region source
 */
FP_PRECISION CPUSolver::getFSRSource(int fsr_id, int energy_group) {

  /* Error checking */
  if (fsr_id >= _num_FSRs)
    log_printf(ERROR_LOG, "Unable to return a source for FSR ID = %d in energy "
               "group %d since the solver only contains FSR with IDs less than "
               "or equal to %d", fsr_id, energy_group, _num_FSRs-1);

  if (fsr_id < 0)
    log_printf(ERROR_LOG, "Unable to return a source for FSR ID = %d "
               "in energy group %d since FSRs do not have negative IDs",
               fsr_id, energy_group);

  if (energy_group-1 >= _num_groups)
    log_printf(ERROR_LOG, "Unable to return a source for FSR ID = %d "
               "in energy group %d since the solver only has %d energy "
               "groups", fsr_id, energy_group, _num_groups);

  if (energy_group <= 0)
    log_printf(ERROR_LOG, "Unable to return a source for FSR ID = %d "
               "in energy group %d since energy groups are greater than "
               "or equal to 1", fsr_id, energy_group);

  return _source(fsr_id,energy_group-1);
}


/**
 * @brief Return a scalar flux array indexed by FSR IDs and energy groups.
 * @details This energy groups are the innermost index, while the FSR ID is
 *         the outermost index.
 * @return an array of flat source region scalar fluxes
 */
FP_PRECISION* CPUSolver::getFSRScalarFluxes() {

  if (_scalar_flux == NULL)
    log_printf(ERROR_LOG, "Unable to returns the Solver's FSR scalar flux array "
               "since it has not yet been allocated in memory");

  return _scalar_flux;
}


/**
 * @brief Return a surface current array indexed by Cmfd Mesh surface IDs
 *        and energy groups.
 * @return an array of Cmfd Mesh cell surface currents
 */
double* CPUSolver::getSurfaceCurrents() {

  if (_surface_currents == NULL)
    log_printf(ERROR_LOG, "Unable to returns the Solver's Cmfd Mesh surface "
               "currents array since it has not yet been allocated in memory");

  return _surface_currents;
}


/**
 * @brief Sets the number of shared memory OpenMP threads to use (>0).
 * @param num_threads the number of threads
 */
void CPUSolver::setNumThreads(int num_threads) {

  if (num_threads <= 0)
    log_printf(ERROR_LOG, "Unable to set the number of threads for the Solver "
               "to %d since it is less than or equal to 0", num_threads);

  _num_threads = num_threads;

#ifdef _OPENMP
  /* Set the number of threads for OpenMP */
  omp_set_num_threads(_num_threads);
#else
  _num_threads = 1;
#endif
}


/**
 * @brief Allocates memory for Track boundary angular flux and leakage
 *        and FSR scalar flux arrays.
 * @details Deletes memory for old flux arrays if they were allocated for a
 *          previous simulation.
 */
void CPUSolver::initializeFluxArrays() {

  /* Delete old flux arrays if they exist */
  if (_boundary_flux != NULL)
    delete [] _boundary_flux;

  if (_boundary_leakage != NULL)
    delete [] _boundary_leakage;

  if (_scalar_flux != NULL)
    delete [] _scalar_flux;

  if (_thread_fsr_flux != NULL)
    delete [] _thread_fsr_flux;

  int size;

  /* Allocate memory for the Track boundary flux and leakage arrays */
  try{

    size = 2 * _tot_num_tracks * _polar_times_groups;
    _boundary_flux = new FP_PRECISION[size];
    _boundary_leakage = new FP_PRECISION[size];

    /* Allocate an array for the FSR scalar flux */
    size = _num_FSRs * _num_groups;
    _scalar_flux = new FP_PRECISION[size];

    /* Allocate a thread local local memory buffer for FSR scalar flux */
    size = _num_groups * _num_threads;
    _thread_fsr_flux = new FP_PRECISION[size];
  }
  catch(std::exception &e) {
    log_printf(ERROR_LOG, "Could not allocate memory for the Solver's fluxes. "
               "Backtrace:%s", e.what());
  }
}


/**
 * @brief Allocates memory for FSR source arrays.
 * @details Deletes memory for old source arrays if they were allocated for a
 *          previous simulation.
 */
void CPUSolver::initializeSourceArrays() {

  /* Delete old sources arrays if they exist */
  if (_fission_sources != NULL)
    delete [] _fission_sources;

  if (_scatter_sources != NULL)
    delete [] _scatter_sources;

  if (_source != NULL)
    delete [] _source;

  if (_old_source != NULL)
    delete [] _old_source;

  if (_reduced_source != NULL)
    delete [] _reduced_source;

  if (_source_residuals != NULL)
    delete [] _source_residuals;

  int size;

  /* Allocate memory for all source arrays */
  try{
    size = _num_FSRs * _num_groups;
    _fission_sources = new FP_PRECISION[size];
    _source = new FP_PRECISION[size];
    _old_source = new FP_PRECISION[size];
    _reduced_source = new FP_PRECISION[size];

    size = _num_threads * _num_groups;
    _scatter_sources = new FP_PRECISION[size];

    size = _num_FSRs;
    _source_residuals = new FP_PRECISION[size];

  }
  catch(std::exception &e) {
    log_printf(ERROR_LOG, "Could not allocate memory for the solver's FSR "
               "sources array. Backtrace:%s", e.what());
  }
}


/**
 * @brief Creates a polar Quadrature object for the solver.
 * @details Deletes memory for old Quadrature if one was allocated for a
 *          previous simulation.
 */
void CPUSolver::initializePolarQuadrature() {

  /* Deletes the old Quadrature if one existed */
  if (_quad != NULL)
    delete _quad;

  _quad = new Quadrature(_quadrature_type, _num_polar);
  _polar_times_groups = _num_groups * _num_polar;
}


/**
 * @brief Builds a linear interpolation table to compute exponentials for
 *        each segment of each Track for each polar angle.
 */
void CPUSolver::buildExpInterpTable() {

  log_printf(INFO_LOG, "Building exponential interpolation table...");

  FP_PRECISION azim_weight;

  _polar_weights = new FP_PRECISION[_num_azim*_num_polar];

  /* Compute the total azimuthal weight for tracks at each polar angle */
#ifdef _OPENMP
  #pragma omp parallel for private(azim_weight) schedule(guided)
#endif
  for (int i=0; i < _num_azim; i++) {

    azim_weight = _azim_weights[i];

    for (int p=0; p < _num_polar; p++)
      _polar_weights(i,p) = azim_weight*_quad->getMultiple(p)*FOUR_PI;
  }

  /* Set size of interpolation table */
  int num_array_values = 10 * sqrt(1./(8.*_source_convergence_thresh*1e-2));
  _exp_table_spacing = 10. / num_array_values;
  _exp_table_size = _two_times_num_polar * num_array_values;
  _exp_table_max_index = _exp_table_size - _two_times_num_polar - 1.;

  log_printf(DEBUG_LOG, "Exponential interpolation table size: %i, max index: %i",
             _exp_table_size, _exp_table_max_index);

  /* Allocate array for the table */
  _exp_table = new FP_PRECISION[_exp_table_size];

  FP_PRECISION expon;
  FP_PRECISION intercept;
  FP_PRECISION slope;

  /* Create exponential linear interpolation table */
  for (int i=0; i < num_array_values; i ++){
    for (int p=0; p < _num_polar; p++){
      expon = exp(- (i * _exp_table_spacing) / _quad->getSinTheta(p));
      slope = - expon / _quad->getSinTheta(p);
      intercept = expon * (1 + (i * _exp_table_spacing)/_quad->getSinTheta(p));
      _exp_table[_two_times_num_polar * i + 2 * p] = slope;
      _exp_table[_two_times_num_polar * i + 2 * p + 1] = intercept;
    }
  }

  /* Compute the reciprocal of the table entry spacing */
  _inverse_exp_table_spacing = 1.0 / _exp_table_spacing;

  return;
}


/**
 * @brief Initializes the FSR volumes and Materials array.
 * @details This method assigns each FSR a unique, monotonically increasing
 *          ID, sets the Material for each FSR, and assigns a volume based on
 *          the cumulative length of all of the segments inside the FSR.
 */
void CPUSolver::initializeFSRs() {

  log_printf(INFO_LOG, "Initializing flat source regions...");

  /* Delete old FSR arrays if they exist */
  if (_FSR_volumes != NULL)
    delete [] _FSR_volumes;

  if (_FSR_materials != NULL)
    delete [] _FSR_materials;

  _FSR_volumes = (FP_PRECISION*)calloc(_num_FSRs, sizeof(FP_PRECISION));
  _FSR_materials = new Material*[_num_FSRs];
#ifdef _OPENMP
  _FSR_locks = new omp_lock_t[_num_FSRs];
#endif

  int num_segments;
  segment* curr_segment;
  segment* segments;
  FP_PRECISION volume;
  CellBasic* cell;
  Material* material;
  Universe* univ_zero = _geometry->getUniverse(0);

  /* Set each FSR's "volume" by accumulating the total length of all Tracks
   * inside the FSR. Loop over azimuthal angles, Tracks and Track segments. */
  for (int i=0; i < _tot_num_tracks; i++) {

    int azim_index = _tracks[i]->getAzimAngleIndex();
    num_segments = _tracks[i]->getNumSegments();
    segments = _tracks[i]->getSegments();

    for (int s=0; s < num_segments; s++) {
      curr_segment = &segments[s];
      volume = curr_segment->_length * _azim_weights[azim_index];
      _FSR_volumes[curr_segment->_region_id] += volume;
    }
  }

  /* Loop over all FSRs to extract FSR material pointers */
#ifdef _OPENMP
  #pragma omp parallel for private(cell, material) schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {

    /* Get the Cell corresponding to this FSR from the geometry */
    cell = _geometry->findCellContainingFSR(r);

    /* Get the Cell's Material and assign it to the FSR */
    material = _geometry->getMaterial(cell->getMaterial());
    _FSR_materials[r] = material;

    log_printf(DEBUG_LOG, "FSR ID = %d has Cell ID = %d and Material ID = %d "
               "and volume = %f", r, cell->getId(),
                _FSR_materials[r]->getUid(), _FSR_volumes[r]);
  }

  /* Loop over all FSRs to initialize OpenMP locks */
#ifdef _OPENMP
  #pragma omp parallel for schedule(guided)
  for (int r=0; r < _num_FSRs; r++)
    omp_init_lock(&_FSR_locks[r]);
#endif

  return;
}


/**
 * @brief Initializes Cmfd object for acceleration prior to source iteration.
 * @details Instantiates a dummy Cmfd object if one was not assigned to
 *          the Solver by the user and initializes FSRs, Materials, fluxes
 *          and the Mesh. This method intializes a global array for the
 *          surface currents.
 */
void CPUSolver::initializeCmfd() {

  /* Call parent class method */
  Solver::initializeCmfd();

  /* Delete old Cmfd Mesh surface currents array it it exists */
  if (_surface_currents != NULL)
    delete [] _surface_currents;

  int size;

  /* Allocate memory for the Cmfd Mesh surface currents array */
  try{

    /* Allocate an array for the Cmfd Mesh surface currents */
    if (_cmfd->getMesh()->getCmfdOn()){
      size = _num_mesh_cells * _cmfd->getNumCmfdGroups() * 8;
      _surface_currents = new double[size];
    }

  }
  catch(std::exception &e) {
    log_printf(ERROR_LOG, "Could not allocate memory for the Solver's Cmfd "
               "Mesh surface currents. Backtrace:%s", e.what());
  }

  if (_cmfd->getMesh()->getCmfdOn()){
#ifdef _OPENMP
    /* Initialize an array of OpenMP locks for each Cmfd Mesh surface */
    _mesh_surface_locks = new omp_lock_t[_cmfd->getMesh()->getNumCells() * 8];

      /* Loop over all mesh cells to initialize OpenMP locks */
      #pragma omp parallel for schedule(guided)
      for (int r=0; r < _num_mesh_cells*8; r++)
          omp_init_lock(&_mesh_surface_locks[r]);
#endif
    }

  return;
}


/**
 * @brief Zero each Track's boundary fluxes for each energy group and polar
 *        angle in the "forward" and "reverse" directions.
 */
void CPUSolver::zeroTrackFluxes() {

#ifdef _OPENMP
  #pragma omp parallel for schedule(guided)
#endif
  for (int t=0; t < _tot_num_tracks; t++) {
    for (int d=0; d < 2; d++) {
      for (int p=0; p < _num_polar; p++) {
        for (int e=0; e < _num_groups; e++) {
          _boundary_flux(t,d,p,e) = 0.0;
        }
      }
    }
  }

  return;
}


/**
 * @brief Set the scalar flux for each FSR and energy group to some value.
 * @param value the value to assign to each FSR scalar flux
 */
void CPUSolver::flattenFSRFluxes(FP_PRECISION value) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {
    for (int e=0; e < _num_groups; e++)
      _scalar_flux(r,e) = value;
  }

  return;
}


 /**
  * @brief Set the Cmfd Mesh surface currents for each Mesh cell and energy
  *        group to zero.
  */
void CPUSolver::zeroSurfaceCurrents() {
#ifdef _OPENMP
  #pragma omp parallel for schedule(guided)
#endif
  for (int r=0; r < _num_mesh_cells; r++) {
    for (int s=0; s < 8; s++) {
      for (int e=0; e < _num_groups; e++)
        _surface_currents(r*8+s,e) = 0.0;
    }
  }

  return;
}


/**
 * @brief Set the source for each FSR and energy group to some value.
 * @param value the value to assign to each FSR source
 */
void CPUSolver::flattenFSRSources(FP_PRECISION value) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {
    for (int e=0; e < _num_groups; e++) {
      _source(r,e) = value;
      _old_source(r,e) = value;
    }
  }

  return;
}


/**
 * @brief Normalizes all FSR scalar fluxes and Track boundary angular
 *        fluxes to the total fission source (times \f$ \nu \f$).
 */
void CPUSolver::normalizeFluxes() {

  FP_PRECISION* nu_sigma_f;
  FP_PRECISION volume;
  FP_PRECISION tot_fission_source;
  FP_PRECISION norm_factor;

  /* Compute total fission source for each FSR, energy group */
#ifdef _OPENMP
  #pragma omp parallel for private(volume, nu_sigma_f) \
    reduction(+:tot_fission_source) schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {

    /* Get pointers to important data structures */
    nu_sigma_f = _FSR_materials[r]->getNuSigmaF();
    volume = _FSR_volumes[r];

    for (int e=0; e < _num_groups; e++)
      _fission_sources(r,e) = nu_sigma_f[e] * _scalar_flux(r,e) * volume;
  }

  /* Compute the total fission source */
  tot_fission_source = pairwise_sum<FP_PRECISION>(_fission_sources,
                                                  _num_FSRs*_num_groups);

  /* Normalize scalar fluxes in each FSR */
  norm_factor = 1.0 / tot_fission_source;

  log_printf(DEBUG_LOG, "Tot. Fiss. Src = %f, Normalization factor = %f",
             tot_fission_source, norm_factor);

#ifdef _OPENMP
  #pragma omp parallel for schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {
    for (int e=0; e < _num_groups; e++)
      _scalar_flux(r,e) *= norm_factor;
  }

  /* Normalize angular boundary fluxes for each Track */
#ifdef _OPENMP
  #pragma omp parallel for schedule(guided)
#endif
  for (int i=0; i < _tot_num_tracks; i++) {
    for (int j=0; j < 2; j++) {
      for (int p=0; p < _num_polar; p++) {
        for (int e=0; e < _num_groups; e++) {
          _boundary_flux(i,j,p,e) *= norm_factor;
        }
      }
    }
  }

  return;
}


/**
 * @brief Computes the total source (fission and scattering) in each FSR.
 * @details This method computes the total source in each FSR based on
 *          this iteration's current approximation to the scalar flux. A
 *          residual for the source with respect to the source compute on
 *          the previous iteration is computed and returned. The residual
 *          is determined as follows:
 *          /f$ res = \sqrt{\frac{\displaystyle\sum \displaystyle\sum
 *                    \left(\frac{Q^i - Q^{i-1}{Q^i}\right)^2}{\# FSRs}}} \f$
 *
 * @return the residual between this source and the previous source
 */
FP_PRECISION CPUSolver::computeFSRSources() {

  int tid;
  Material* material;
  FP_PRECISION scatter_source;
  FP_PRECISION fission_source;
  FP_PRECISION* nu_sigma_f;
  FP_PRECISION* sigma_s;
  FP_PRECISION* sigma_t;
  FP_PRECISION* chi;

  FP_PRECISION source_residual = 0.0;

  FP_PRECISION inverse_k_eff = 1.0 / _k_eff;

  /* For all FSRs, find the source */
#ifdef _OPENMP
  #pragma omp parallel for private(tid, material, nu_sigma_f, chi, \
    sigma_s, sigma_t, fission_source, scatter_source) schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {
#ifdef _OPENMP
    tid = omp_get_thread_num();
#else
    tid = 0;
#endif
    material = _FSR_materials[r];
    nu_sigma_f = material->getNuSigmaF();
    chi = material->getChi();
    sigma_s = material->getSigmaS();
    sigma_t = material->getSigmaT();

    /* Initialize the source residual to zero */
    _source_residuals[r] = 0.;

    /* Compute fission source for each group */
    if (material->isFissionable()) {
      for (int e=0; e < _num_groups; e++)
        _fission_sources(r,e) = _scalar_flux(r,e) * nu_sigma_f[e];

        fission_source = pairwise_sum<FP_PRECISION>(&_fission_sources(r,0),
                                                     _num_groups);
        fission_source *= inverse_k_eff;
    }

    else
      fission_source = 0.0;

    /* Compute total scattering source for group G */
    for (int G=0; G < _num_groups; G++) {
      scatter_source = 0;

      for (int g=0; g < _num_groups; g++)
        _scatter_sources(tid,g) = sigma_s[G*_num_groups+g] * _scalar_flux(r,g);

        scatter_source=pairwise_sum<FP_PRECISION>(&_scatter_sources(tid,0),
                                                   _num_groups);

      /* Set the total source for FSR r in group G */
      _source(r,G) = (fission_source * chi[G] + scatter_source) *
                      ONE_OVER_FOUR_PI;

      _reduced_source(r,G) = _source(r,G) / sigma_t[G];

      /* Compute the norm of residual of the source in the FSR */
      if (fabs(_source(r,G)) > 1E-10)
        _source_residuals[r] += pow((_source(r,G) - _old_source(r,G))
                                / _source(r,G), 2);

      /* Update the old source */
      _old_source(r,G) = _source(r,G);
    }
  }

  /* Sum up the residuals from each FSR */
  source_residual = pairwise_sum<FP_PRECISION>(_source_residuals, _num_FSRs);
  source_residual = sqrt(source_residual / (_num_FSRs * _num_groups));

  return source_residual;
}


/**
 * @brief Compute \f$ k_{eff} \f$ from the total fission and absorption rates.
 * @details This method computes the current approximation to the
 *          multiplication factor on this iteration as follows:
 *          \f$ k_{eff} = \frac{\displaystyle\sum \displaystyle\sum \nu
 *                        \Sigma_f \Phi V}{\displaystyle\sum
 *                        \displaystyle\sum \Sigma_a \Phi V} \f$
 */
void CPUSolver::computeKeff() {

  int tid;
  Material* material;
  FP_PRECISION* sigma_a;
  FP_PRECISION* nu_sigma_f;
  FP_PRECISION volume;

  FP_PRECISION tot_abs = 0.0;
  FP_PRECISION tot_fission = 0.0;

  FP_PRECISION* FSR_rates = new FP_PRECISION[_num_FSRs];
  FP_PRECISION* group_rates = new FP_PRECISION[_num_threads * _num_groups];

  /* Loop over all FSRs and compute the volume-weighted absorption rates */
#ifdef _OPENMP
  #pragma omp parallel for private(tid, volume, \
    material, sigma_a) schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {
#ifdef _OPENMP
    tid = omp_get_thread_num() * _num_groups;
#else
    tid = 0;
#endif
    volume = _FSR_volumes[r];
    material = _FSR_materials[r];
    sigma_a = material->getSigmaA();

    for (int e=0; e < _num_groups; e++)
      group_rates[tid+e] = sigma_a[e] * _scalar_flux(r,e);

    FSR_rates[r]=pairwise_sum<FP_PRECISION>(&group_rates[tid], _num_groups);
    FSR_rates[r] *= volume;
  }

  /* Reduce absorption rates across FSRs */
  tot_abs = pairwise_sum<FP_PRECISION>(FSR_rates, _num_FSRs);

  /* Loop over all FSRs and compute the volume-weighted fission rates */
#ifdef _OPENMP
  #pragma omp parallel for private(tid, volume, \
    material, nu_sigma_f) schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {
#ifdef _OPENMP
    tid = omp_get_thread_num() * _num_groups;
#else
    tid = 0;
#endif
    volume = _FSR_volumes[r];
    material = _FSR_materials[r];
    nu_sigma_f = material->getNuSigmaF();

    for (int e=0; e < _num_groups; e++)
      group_rates[tid+e] = nu_sigma_f[e] * _scalar_flux(r,e);

    FSR_rates[r]=pairwise_sum<FP_PRECISION>(&group_rates[tid], _num_groups);
    FSR_rates[r] *= volume;
  }

  /* Reduce fission rates across FSRs */
  tot_fission = pairwise_sum<FP_PRECISION>(FSR_rates, _num_FSRs);

  /** Reduce leakage array across Tracks, energy groups, polar angles */
  int size = 2 * _tot_num_tracks * _polar_times_groups;
  _leakage = pairwise_sum<FP_PRECISION>(_boundary_leakage, size) * 0.5;

  _k_eff = tot_fission / (tot_abs + _leakage);

  log_printf(DEBUG_LOG, "abs = %f, fission = %f, leakage = %f, k_eff = %f",
             tot_abs, tot_fission, _leakage, _k_eff);

  delete [] FSR_rates;
  delete [] group_rates;

  return;
}


/**
 * @brief This method performs one transport sweep of all azimuthal angles,
 *        Tracks, Track segments, polar angles and energy groups.
 * @details The method integrates the flux along each Track and updates the
 *          boundary fluxes for the corresponding output Track, while updating
 *          the scalar flux in each flat source region.
 */
void CPUSolver::transportSweep() {

  int tid;
  int min_track, max_track;
  Track* curr_track;
  int azim_index;
  int num_segments;
  segment* curr_segment;
  segment* segments;
  FP_PRECISION* track_flux;

  log_printf(DEBUG_LOG, "Transport sweep with %d OpenMP threads", _num_threads);

  /* Initialize flux in each FSr to zero */
  flattenFSRFluxes(0.0);

  if (_cmfd->getMesh()->getCmfdOn())
    zeroSurfaceCurrents();

  /* Loop over azimuthal angle halfspaces */
  for (int i=0; i < 2; i++) {

    /* Compute the minimum and maximum Track IDs corresponding to
     * this azimuthal angular halfspace */
    min_track = i * (_tot_num_tracks / 2);
    max_track = (i + 1) * (_tot_num_tracks / 2);

    /* Loop over each thread within this azimuthal angle halfspace */
#ifdef _OPENMP
    #pragma omp parallel for private(curr_track, azim_index, num_segments, \
      curr_segment, segments, track_flux, tid) schedule(guided)
#endif
    for (int track_id=min_track; track_id < max_track; track_id++) {
#ifdef _OPENMP
      tid = omp_get_thread_num();
#else
      tid = 0;
#endif

      /* Initialize local pointers to important data structures */
      curr_track = _tracks[track_id];
      azim_index = curr_track->getAzimAngleIndex();
      num_segments = curr_track->getNumSegments();
      segments = curr_track->getSegments();
      track_flux = &_boundary_flux(track_id,0,0,0);

      /* Loop over each Track segment in forward direction */
      for (int s=0; s < num_segments; s++) {
        curr_segment = &segments[s];
        scalarFluxTally(curr_segment, azim_index, track_flux,
                        &_thread_fsr_flux(tid),true);
      }

      /* Transfer boundary angular flux to outgoing Track */
      transferBoundaryFlux(track_id, azim_index, true, track_flux);

      /* Loop over each Track segment in reverse direction */
      track_flux += _polar_times_groups;

      for (int s=num_segments-1; s > -1; s--) {
        curr_segment = &segments[s];
        scalarFluxTally(curr_segment, azim_index, track_flux,
                        &_thread_fsr_flux(tid),false);
      }

      /* Transfer boundary angular flux to outgoing Track */
      transferBoundaryFlux(track_id, azim_index, false, track_flux);
    }
  }

  return;
}


/**
 * @brief Computes the contribution to the FSR scalar flux from a Track segment.
 * @details This method integrates the angular flux for a Track segment across
 *          energy groups and polar angles, and tallies it into the FSR
 *          scalar flux, and updates the Track's angular flux.
 * @param curr_segment a pointer to the Track segment of interest
 * @param azim_index a pointer to the azimuthal angle index for this segment
 * @param track_flux a pointer to the Track's angular flux
 * @param fsr_flux a pointer to the temporary FSR flux buffer
 * @param fwd
 */
void CPUSolver::scalarFluxTally(segment* curr_segment,
                                int azim_index,
                                FP_PRECISION* track_flux,
                                FP_PRECISION* fsr_flux,
                                bool fwd){
#ifdef _OPENMP
  int tid = omp_get_thread_num();
#else
  int tid = 0;
#endif
  int fsr_id = curr_segment->_region_id;
  FP_PRECISION length = curr_segment->_length;
  FP_PRECISION* sigma_t = curr_segment->_material->getSigmaT();

  /* The change in angular flux along this Track segment in the FSR */
  FP_PRECISION delta_psi;
  FP_PRECISION exponential;

  /* Set the FSR scalar flux buffer to zero */
  memset(fsr_flux, 0.0, _num_groups * sizeof(FP_PRECISION));

  /* Loop over energy groups */
  for (int e=0; e < _num_groups; e++) {

    /* Loop over polar angles */
    for (int p=0; p < _num_polar; p++){
      exponential = computeExponential(sigma_t[e], length, p);
      delta_psi = (track_flux(p,e)-_reduced_source(fsr_id,e))*exponential;
      fsr_flux[e] += delta_psi * _polar_weights[p];
      track_flux(p,e) -= delta_psi;
    }
  }

  if (_cmfd->getMesh()->getCmfdOn()){
    if (curr_segment->_mesh_surface_fwd != -1 && fwd){

      int pe = 0;

      /* Atomically increment the Cmfd Mesh surface current from the
       * temporary array using mutual exclusion locks */
#ifdef _OPENMP
      omp_set_lock(&_mesh_surface_locks[curr_segment->_mesh_surface_fwd]);
#endif

      /* Loop over energy groups */
      for (int e = 0; e < _num_groups; e++) {

        /* Loop over polar angles */
        for (int p = 0; p < _num_polar; p++){

          /* Increment current (polar and azimuthal weighted flux, group) */
          _surface_currents(curr_segment->_mesh_surface_fwd,e) +=
                                        track_flux(p,e)*_polar_weights[p]/2.0;
          pe++;
        }
      }

#ifdef _OPENMP
      /* Release Cmfd Mesh surface mutual exclusion lock */
      omp_unset_lock(&_mesh_surface_locks[curr_segment->_mesh_surface_fwd]);
#endif

    }
    else if (curr_segment->_mesh_surface_bwd != -1 && !fwd){

      int pe = 0;

#ifdef _OPENMP
      /* Atomically increment the Cmfd Mesh surface current from the
       * temporary array using mutual exclusion locks */
      omp_set_lock(&_mesh_surface_locks[curr_segment->_mesh_surface_bwd]);
#endif

      /* Loop over energy groups */
      for (int e = 0; e < _num_groups; e++) {

        /* Loop over polar angles */
        for (int p = 0; p < _num_polar; p++){

          /* Increment current (polar and azimuthal weighted flux, group) */
          _surface_currents(curr_segment->_mesh_surface_bwd,e) +=
                                        track_flux(p,e)*_polar_weights[p]/2.0;
          pe++;
        }
      }

#ifdef _OPENMP
      /* Release Cmfd Mesh surface mutual exclusion lock */
      omp_unset_lock(&_mesh_surface_locks[curr_segment->_mesh_surface_bwd]);
#endif
    }
  }

#ifdef _OPENMP
  /* Atomically increment the FSR scalar flux from the temporary array */
  omp_set_lock(&_FSR_locks[fsr_id]);
#endif
  {
    for (int e=0; e < _num_groups; e++)
      _scalar_flux(fsr_id,e) += fsr_flux[e];
  }
#ifdef _OPENMP
  omp_unset_lock(&_FSR_locks[fsr_id]);
#endif

  return;
}


/**
 * @brief Computes the exponential term in the transport equation for a
 *        Track segment.
 * @details This method computes \f$ 1 - exp(-l\Sigma^T_g/sin(\theta_p)) \f$
 *          for a segment with total group cross-section and for some polar
 *          angle. This method uses either a linear interpolation table
 *          (default) or the exponential intrinsic exp(...) function if
 *          requested by the user through a call to the
 *          Solver::useExponentialIntrinsic()  routine.
 * @param sigma_t the total group cross-section at this energy
 * @param length the length of the Track segment projected in the xy-plane
 * @param p the polar angle index
 * @return the evaluated exponential
 */
FP_PRECISION CPUSolver::computeExponential(FP_PRECISION sigma_t,
                                           FP_PRECISION length, int p) {

  FP_PRECISION exponential;
  FP_PRECISION tau = sigma_t * length;

  /* Evaluate the exponential using the lookup table - linear interpolation */
  if (_interpolate_exponential) {
    int index;
    index = round_to_int(tau * _inverse_exp_table_spacing);
    index *= _two_times_num_polar;
    exponential = (1. - (_exp_table[index+2 * p] * tau +
                  _exp_table[index + 2 * p +1]));
  }

  /* Evalute the exponential using the intrinsic exp(...) function */
  else {
    FP_PRECISION sintheta = _quad->getSinTheta(p);
    exponential = 1.0 - exp(- tau / sintheta);
  }

  return exponential;
}


/**
 * @brief Updates the boundary flux for a Track given boundary conditions.
 * @details For reflective boundary conditions, the outgoing boundary flux
 *          for the Track is given to the reflecting Track. For vacuum
 *          boundary conditions, the outgoing flux tallied as leakage.
 * @param track_id the ID number for the Track of interest
 * @param azim_index a pointer to the azimuthal angle index for this segment
 * @param direction the Track direction (forward - true, reverse - false)
 * @param track_flux a pointer to the Track's outgoing angular flux
 */
void CPUSolver::transferBoundaryFlux(int track_id,
                                     int azim_index,
                                     bool direction,
                                     FP_PRECISION* track_flux) {
  int start;
  int bc;
  FP_PRECISION* track_leakage;
  int track_out_id;

  /* Extract boundary conditions for this Track and the pointer to the
   * outgoing reflective Track, and index into the leakage array */

  /* For the "forward" direction */
  if (direction) {
    start = _tracks[track_id]->isReflOut() * _polar_times_groups;
    bc = (int)_tracks[track_id]->getBCOut();
    track_leakage = &_boundary_leakage(track_id,0);
    track_out_id = _tracks[track_id]->getTrackOut()->getUid();
  }

  /* For the "reverse" direction */
  else {
    start = _tracks[track_id]->isReflIn() * _polar_times_groups;
    bc = (int)_tracks[track_id]->getBCIn();
    track_leakage = &_boundary_leakage(track_id,_polar_times_groups);
    track_out_id = _tracks[track_id]->getTrackIn()->getUid();
  }

  FP_PRECISION* track_out_flux = &_boundary_flux(track_out_id,0,0,start);

  /* Loop over polar angles and energy groups */
  for (int e=0; e < _num_groups; e++) {
    for (int p=0; p < _num_polar; p++) {
      track_out_flux(p,e) = track_flux(p,e) * bc;
      track_leakage(p,e) = track_flux(p,e) *
                            _polar_weights(azim_index,p) * (!bc);
    }
  }
}


/**
 * @brief Add the source term contribution in the transport equation to
 *        the FSR scalar flux.
 */
void CPUSolver::addSourceToScalarFlux() {

  FP_PRECISION volume;
  FP_PRECISION* sigma_t;

  /* Add in source term and normalize flux to volume for each FSR */
  /* Loop over FSRs, energy groups */
#ifdef _OPENMP
  #pragma omp parallel for private(volume, sigma_t) schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {

    volume = _FSR_volumes[r];
    sigma_t = _FSR_materials[r]->getSigmaT();

      for (int e=0; e < _num_groups; e++) {
        _scalar_flux(r,e) *= 0.5;
        _scalar_flux(r,e) = FOUR_PI * _reduced_source(r,e) +
                            (_scalar_flux(r,e) / (sigma_t[e] * volume));
    }
  }

  return;
}


/**
 * @brief Computes the volume-weighted, energy integrated fission rate in
 *        each FSR and stores them in an array indexed by FSR ID.
 * @details This is a helper method for SWIG to allow users to retrieve
 *          FSR fission rates as a NumPy array. An example of how this method 
 *          can be called from Python is as follows:
 *
 * @code
 *          num_FSRs = geometry.getNumFSRs()
 *          fission_rates = solver.computeFSRFissionRates(num_FSRs)
 * @endcode
 *
 * @param fission_rates an array to store the fission rates (implicitly passed
 *                      in as a NumPy array from Python)
 * @param num_FSRs the number of FSRs passed in from Python
 */
void CPUSolver::computeFSRFissionRates(double* fission_rates, int num_FSRs) {

  log_printf(INFO_LOG, "Computing FSR fission rates...");

  FP_PRECISION* sigma_f;

  FP_PRECISION* scalar_flux = getFSRScalarFluxes();

  /* Initialize fission rates to zero */
  for (int r=0; r < _num_FSRs; r++)
    fission_rates[r] = 0.0;

  /* Loop over all FSRs and compute the volume-weighted fission rate */
#ifdef _OPENMP
  #pragma omp parallel for private (sigma_f) schedule(guided)
#endif
  for (int r=0; r < _num_FSRs; r++) {
    sigma_f = _FSR_materials[r]->getSigmaF();

    for (int e=0; e < _num_groups; e++)
      fission_rates[r] += sigma_f[e] * _scalar_flux(r,e);
  }

  return;
}
