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

#ifndef _LOCALCOORDS_H_
#define _LOCALCOORDS_H_       1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file LocalCoords.h
 * @brief The LocalCoords class.
 * @date January 25, 2012
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifdef __cplusplus
  #include "Point.h"
  #include "Universe.h"
#endif

/**
 * @enum coordType
 * @brief The type of Universe level on which the LocalCoords reside
 */
enum coordType {
  /** A Universe level coordinate type */
  UNIV,

  /** A Lattice level coordinate type */
  LAT
};


/**
 * @class LocalCoords LocalCoords.h "openmoc/src/host/LocalCoords.h"
 * @brief The LocalCoords represents a set of local coordinates on some
 *        level of nested Universes making up the geometry.
 */
class LocalCoords {

private:
  /** The local coordinate type (UNIV or LAT) */
  coordType _type;

  /** The ID of the Universe within which this LocalCoords resides */
  int _universe;

  /** The ID of the Cell within which this LocalCoords resides */
  int _cell;

  /** The ID of the Lattice within which this LocalCoords resides */
  int _lattice;

  /** The first index of the Lattice cell within which this LocalCoords
   *  resides */
  int _lattice_x;

  /** The second index of the Lattice cell within which this LocalCoords
   *  resides */
  int _lattice_y;

  /** A Point representing the 2D coordinates of this LocalCoords */
  Point _coords;

  /** A pointer to the LocalCoords at the next lower nested Universe level */
  LocalCoords* _next;

  /** A pointer to the LocalCoords at the next higher nested Universe level */
  LocalCoords* _prev;

public:
  LocalCoords(double x, double y);
  virtual ~LocalCoords();
  coordType getType();
  int getUniverse() const;
  int getCell() const;
  int getLattice() const;
  int getLatticeX() const;
  int getLatticeY() const;
  double getX() const;
  double getY() const;
  Point* getPoint();
  LocalCoords* getNext() const;
  LocalCoords* getPrev() const;

  void setType(coordType type);
  void setUniverse(int universe);
  void setCell(int cell);
  void setLattice(int lattice);
  void setLatticeX(int lattice_x);
  void setLatticeY(int lattice_y);
  void setX(double x);
  void setY(double y);
  void setNext(LocalCoords *next);
  void setPrev(LocalCoords* coords);

  LocalCoords* getLowestLevel();
  void adjustCoords(double delta_x, double delta_y);
  void updateMostLocal(Point* point);
  void prune();
  void copyCoords(LocalCoords* coords);
  std::string toString();
};


#endif /* LOCALCOORDS_H_ */
