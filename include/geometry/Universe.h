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

#ifndef _UNIVERSE_H_
#define _UNIVERSE_H_

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file Universe.h
 * @brief The Universe class.
 * @date January 9, 2012
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifdef __cplusplus
  #include <map>
  #include <vector>
  #include "Cell.h"
  #include "LocalCoords.h"
#endif


/** Error threshold for determining how close to the boundary of a Lattice cell
 * a Point needs to be to be considered on it */
#define ON_LATTICE_CELL_THRESH 1E-12


/** Distance a Point is moved to cross over a Surface into a new Cell during
 * Track segmentation */
#define TINY_MOVE 1E-10


class LocalCoords;
class Cell;
class CellFill;
class CellBasic;


int universe_id();


/**
 * @enum universeType
 * @brief The type of universe
 */
enum universeType{

  /** A simple non-repeating Universe */
  SIMPLE,

  /** A collection of Universes in a rectangular Lattice */
  LATTICE
};


/**
 * @class Universe Universe.h "src/Universe.h"
 * @brief A Universe represents an unbounded space in the 2D xy-plane.
 * @details A Universe contains cell which are bounded subspaces in the 2D
 *          xy-plane and which together form the Universe. Universes allow
 *          for complex, repeating (i.e. lattices) geometries to be simply
 *          represented with as few data structures as possible.
 */
class Universe {

protected:

  /** A static counter for the number of Universes */
  static int _n;

  /** A monotonically increasing unique ID for each Universe created */
  int _uid;

  /** A user-defined id for each Universe created */
  int _id;

  /** The type of Universe (ie, SIMLE or LATTICE) */
  universeType _type;

  /** A collection of Cell IDs and Cell pointers */
  std::map<int, Cell*> _cells;

  /** The coordinates of the origin for the Universe */
  Point _origin;

  /** A collection of Cell IDs and their corresponding flat source region IDs.
   *  This helps for computing FSR IDs and building FSR maps for plotting
   *  FSR-based quantities such as the scalar flux and pin powers. */
  std::map<int, int> _region_map;

  /** A boolean representing whether or not this Universe contains a Material
   *  with a non-zero fission cross-section and is fissionable */
  bool _fissionable;

public:

  Universe(const int id);
  virtual ~Universe();

  void addCell(Cell* cell);

  Cell* getCell(int cell_id);
  CellFill* getCellFill(int cell_id);
  CellBasic* getCellBasic(int cell_id);
  std::map<int, Cell*> getCells() const;
  int getUid() const;
  int getId() const;
  universeType getType();
  int getNumCells() const;
  int getFSR(int cell_id);
  Point* getOrigin();
  std::vector<int> getMaterialIds();
  std::vector<int> getNestedUniverseIds();
  void getCellIds(int* cell_ids, int num_cells);
  bool isFissionable();

  void setType(universeType type);
  void setOrigin(Point* origin);
  void setFissionability(bool fissionable);

  Cell* findCell(LocalCoords* coords, std::map<int, Universe*> universes);
  int computeFSRMaps();
  void subdivideCells();
  std::string toString();
  void printString();

  Universe* clone();
};


/**
 * @class Lattice Universe.h "src/Universe.h"
 * @brief Represents a repeating 2D Lattice of Universes.
 */
class Lattice: public Universe {

private:

  /** The number of Lattice cells along the x-axis */
  int _num_x;

  /** The number of Lattice cells along the y-axis */
  int _num_y;

  /** The width of each Lattice cell (cm) along the x-axis */
  double _width_x;

  /** The width of each Lattice cell (cm) along the y-axis */
  double _width_y;

  /** A container of Universes ? */
  std::vector< std::vector< std::pair<int, Universe*> > > _universes;

  /** A container of the number of FSRs in each Lattice cell */
  std::vector< std::vector< std::pair<int, int> > > _region_map;

public:

  Lattice(const int id, const double width_x, const double width_y);
  virtual ~Lattice();

  int getNumX() const;
  int getNumY() const;
  Point* getOrigin();
  std::vector< std::vector< std::pair<int, Universe*> > >
                                           getUniverses() const;
  Universe* getUniverse(int lattice_x, int lattice_y) const;
  double getWidthX() const;
  double getWidthY() const;
  int getFSR(int lat_x, int lat_y);
  std::vector<int> getNestedUniverseIds();

  void setLatticeCells(int num_x, int num_y, int* universes);
  void setUniversePointer(Universe* universe);

  bool withinBounds(Point* point);
  Cell* findCell(LocalCoords* coords, std::map<int, Universe*> universes);
  Cell* findNextLatticeCell(LocalCoords* coords, double angle,
                            std::map<int, Universe*> universes);
  int computeFSRMaps();

  std::string toString();
  void printString();
};

#endif /* UNIVERSE_H_ */

