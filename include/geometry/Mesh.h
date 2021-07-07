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

#ifndef _MESH_H_
#define _MESH_H_      1

#include "Basic.h"

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
// Be ralated to the structured meshes (grids) and the un-structured meshes (grids)
//

/**
 * @file Mesh.h
 * @brief The Mesh class.
 * @date October 14, 2013
 * @author Sam Shaner, MIT, Course 22 (shaner@mit.edu)
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <string>
#include <map>
#include <utility>
#include <sstream>
#include "common/log.h"
#include "LocalCoords.h"
#include "Surface.h"
#include "Material.h"


/**
 * @enum solveType
 * @brief The type of solution method.
 */
enum solveType {

  /** Diffusion method */
  DIFFUSION,

  /** Method of Characteristics transport */
  MOC
};


/**
 * @enum fluxType
 * @brief The type of scalar flux.
*/
enum fluxType {

  /** The primal scalar flux */
  PRIMAL,

  /** The updated primal scalar flux */
  PRIMAL_UPDATE,

  /** The adjoint scalar flux */
  ADJOINT
};



/**
 * @class Mesh Mesh.h "src/Mesh.h"
 * @brief Represents a cartesian mesh for Coarse Mesh Finite Difference
 *        Diffusion acceleration.
 */
class Mesh {

private:

  /** The length of the Mesh along the x direction */
  double _length_x;

  /** The length of the Mesh along the y direction */
  double _length_y;

  /** Cmfd nested universee level (>0) */
  int _mesh_level;

  /** The number of cells in the x direction */
  int _num_x;

  /** The nu8mber of cells in the y direction */
  int _num_y;

  /** The number of groups */
  int _num_groups;

  /** The number of surface current values */
  int _num_currents;

  /** The number of FSRs */
  int _num_fsrs;

  /** An array of boundaryTypes */
  boundaryType* _boundaries;

  /** An array of Mesh cell volumes (areas) */
  double* _volumes;

  /** An array of Mesh surface currents */
  double* _currents;

  /** A std::vector of std::vectors of FSR IDs in each Mesh cell */
  std::vector< std::vector<int> > _cell_fsrs;

  /** CMFD in use flag */
  bool _cmfd_on;

  /** CMFD acceleration flag */
  bool _acceleration;

  /** Relaxation factor on d_tilde */
  double _relax_factor;

  /** A std::map of fluxes mapped onto the mesh */
  std::map<fluxType, double*> _fluxes;

  /** Array of Materials */
  Material** _materials;

  /** An array of FSR bounds */
  int* _fsr_indices;

  /** An array of lengths of each Mesh cell in x direction */
  double* _lengths_x;

  /** An array of lengths of each Mesh cell in y direction */
  double* _lengths_y;

  /** An array of cell bounds in x direction */
  double* _bounds_x;

  /** An array of cell bounds in y direction */
  double* _bounds_y;

  /** A switch to toggle optically thick diffusion correction factor */
  bool _optically_thick;

  /** The solution method (DIFFUSION or MOC) */
  solveType _solve_method;


public:

  Mesh(solveType solve_type=MOC, bool cmfd_on=false,
       double relax_factor=0.6, int mesh_level=-1);
  virtual ~Mesh();

  void initialize();
  void setFSRBounds();
  void setCellBounds();

  /* Get mesh parameters */
  double getLengthX();
  double getLengthY();
  int getCellsX();
  int getCellsY();
  int getNumCells();
  boundaryType getBoundary(int side);
  int getNumCurrents();
  double getFlux(int cell_id, int group, fluxType flux_type=PRIMAL_UPDATE);
  std::vector<std::vector<int> >* getCellFSRs();
  Material** getMaterials();
  double* getVolumes();
  double* getFluxes(fluxType flux_type);
  double* getLengthsX();
  double* getLengthsY();
  double* getCurrents();
  int getMeshLevel();

  /* Set mesh parameters */
  void setLengthX(double length_x);
  void setLengthY(double length_y);
  void setCellLengthX(int cell_num, double length_x);
  void setCellLengthY(int cell_num, double length_y);
  void setCellsX(int cells_x);
  void setCellsY(int cells_y);
  void setSurfaceCurrents(double* surface_currents);
  void setVolume(double volume, int cell_num);
  void setMeshLevel(int cmfd_level);

  /* Set general problem specs */
  void setNumGroups(int num_groups);
  void setNumFSRs(int num_fsrs);
  void setAcceleration(bool accel);
  void setOpticallyThick(bool thick);
  void setRelaxFactor(double relax_factor);

  /* Get generation problem specs */
  int getNumGroups();
  int getNumFSRs();
  bool getCmfdOn();
  bool getAcceleration();
  bool getOpticallyThick();
  double getRelaxFactor();
  solveType getSolveType();

  /* worker functions */
  int findMeshCell(double x, double y);
  int findMeshSurface(int fsr_id, LocalCoords* coord);
  void printCurrents();
  void splitCorners();
  void setBoundary(int side, boundaryType boundary);
  int getCellNext(int cell_num, int surface_id);
  int findCellId(LocalCoords* coord);
  void initializeMaterialsMOC();
  void initializeMaterialsDiffusion(std::map<int, Material*>* materials,
                                    int* fsrs_to_mats);
  void initializeSurfaceCurrents();
  void initializeFlux();
};

#endif /* MESH_H_ */
