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

#ifndef _SURFACE_H_
#define _SURFACE_H_     1

#include "Basic.h"

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file Surface.h
 * @details The Surface class and subclasses.
 * @date January 9, 2012
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifdef __cplusplus
  #include <limits>
  #include "Cell.h"
  #include "LocalCoords.h"
#endif

/** Error threshold for determining how close a point needs to be to a surface
 * to be considered on it */
#define ON_SURFACE_THRESH 1E-12


/* Define for compiler */
class LocalCoords;
class Cell;


int surf_id();


/**
 * @enum surfaceType
 * @brief The types of surfaces supported by OpenMOC.
 */
enum surfaceType {
  /** A general plane perpendicular to the 2D xy plane */
  PLANE,

  /** A circle with axis parallel to the z-axis */
  CIRCLE,

  /** A plane perpendicular to the x-axis */
  XPLANE,

  /** A plane perpendicular to the y-axis */
  YPLANE,

  /** A plane perpendicular to the z-axis */
  ZPLANE,

  /** A generalized quadratic surface */
  QUADRATIC
};


/**
 * @enum boundaryType
 * @brief The types of boundary conditions supported by OpenMOC for surfaces.
 */
enum boundaryType {
  /** A vacuum boundary condition */
  VACUUM,

  /** A reflective boundary condition */
  REFLECTIVE,

  /** A zero flux boundary condition */
  ZERO_FLUX,

  /** No boundary type (typically an interface between flat source regions) */
  BOUNDARY_NONE
};



/**
 * @class Surface Surface.h "src/Surface.h"
 * @brief Represents a general surface in the 2D xy-plane
 * @details The Surface class and its subclasses are used to define the
 *          geometry for an OpenMOC simulation using a constructive solid
 *          geometry (CSG) formalism. Surfaces are used during ray tracing
 *          of charateristic tracks across the geometry.
 */
class Surface {

protected:

  /** A static counter fo the number of surfaces in a simulation */
  static int _n;

  /** A monotonically increasing unique ID for each surface created */
  int _uid;

  /** A user-defined id for each surface created */
  int _id;

  /** The type of surface (ie, XPLANE, CIRCLE, etc) */
  surfaceType _surface_type;

  /** The type of boundary condition to be used for this surface
   *  (ie, VACUUM or REFLECTIVE) */
  boundaryType _boundary_type;

public:
  Surface(const int id=0);
  virtual ~Surface();

  int getUid() const;
  int getId() const;
  surfaceType getSurfaceType();
  boundaryType getBoundaryType();

  /**
   * @brief Returns the minimum x value on this Surface.
   * @return the minimum x value
   */
  virtual double getXMin() =0;

  /**
   * @brief Returns the maximum x value on this Surface.
   * @return the maximum x value
   */
  virtual double getXMax() =0;

  /**
   * @brief Returns the minimum y value on this Surface.
   * @return the minimum y value
   */
  virtual double getYMin() =0;

  /**
   * @brief Returns the maximum y value on this Surface.
   * @return the maximum y value
   */
  virtual double getYMax() =0;

  void setBoundaryType(const boundaryType boundary_type);

  /**
   * @brief Evaluate a Point using the Surface's potential equation.
   * @details This method returns the values \f$ f(x,y) \f$ for the potential
   *          function \f$f\f$ representing this Surface.
   * @param point a pointer to the Soint of interest
   * @return the value of Point in the Plane's potential equation.
   */
  virtual double evaluate(const Point* point) const =0;

  /**
   * @brief Finds the intersection Point with this Surface from a given
   *        Point and trajectory defined by an angle.
   * @param point pointer to the Point of interest
   * @param angle the angle defining the trajectory in radians
   * @param points pointer to a Point to store the intersection Point
   * @return the number of intersection Points (0 or 1)
   */
  virtual int intersection(Point* point, double angle, Point* points) =0;

  bool isPointOnSurface(Point* point);
  bool isCoordOnSurface(LocalCoords* coord);
  double getMinDistance(Point* point, double angle, Point* intersection);

  /**
   * @brief Converts this Surface's attributes to a character array.
   * @details The character array returned conatins the type of Surface (ie,
   *          PLANE) and the coefficients in the potential equation.
   * @return a character array of this Surface's attributes
   */
  virtual std::string toString() =0;

  /**
   * @brief Prints a string representation of all of the Surface's objects to
   *        the console.
   */
  virtual void printString() =0;
};


/**
 * @class Plane Surface.h "src/Surface.h"
 * @brief Represents a Plane perpendicular to the xy-plane.
 */
class Plane: public Surface {

protected:

  /** The coefficient for the linear term in x */
  double _a;

  /** The coefficient for the linear term in y */
  double _b;

  /** The constant offset */
  double _c;

  /** The Plane is a friend of class Surface */
  friend class Surface;

  /** The Plane is a friend of class Circle */
  friend class Circle;

public:

  Plane(const double A, const double B, const double C, const int id=0);

  double getXMin();
  double getXMax();
  double getYMin();
  double getYMax();

  double evaluate(const Point* point) const;
  int intersection(Point* point, double angle, Point* points);

  std::string toString();
  void printString();
};


/**
 * @class XPlane Surface.h "src/Surface.h"
 * @brief Represents a Plane perpendicular to the x-axis.
 */
class XPlane: public Plane {

private:

  /** The location of the XPlane along the x-axis */
  double _x;

public:
  XPlane(const double x, const int id=0);

  void setX(const double x);

  double getX();
  double getXMin();
  double getXMax();
  double getYMin();
  double getYMax();

  std::string toString();
};


/**
 * @class YPlane Surface.h "src/Surface.h"
 * @brief Represents a Plane perpendicular to the y-axis.
 */
class YPlane: public Plane {

private:

  /** The location of the YPlane along the y-axis */
  double _y;

public:
  YPlane(const double y, const int id=0);

  void setY(const double y);

  double getY();
  double getXMin();
  double getXMax();
  double getYMin();
  double getYMax();

  std::string toString();
  void printString();
};


/**
 * @class ZPlane Surface.h "src/Surface.h"
 * @brief Represents a Plane perpendicular to the z-axis.
 */
class ZPlane: public Plane {

private:

  /** The location of the ZPlane along the z-axis */
  double _z;

public:
  ZPlane(const double z, const int id=0);

  void setZ(const double z);

  double getZ();
  double getXMin();
  double getXMax();
  double getYMin();
  double getYMax();

  std::string toString();
  void printString();
};


/**
 * @class Circle Surface.h "src/Surface.h"
 * @brief Represents a Circle in the xy-plane.
 */
class Circle: public Surface {

private:

  /** A 2D point for the Circle's center */
  Point _center;

  /** The Circle's radius */
  double _radius;

  /** The coefficient of the x-squared term */
  double _a;

  /** The coefficient of the y-squared term */
  double _b;

  /** The coefficient of the linear term in x */
  double _c;

  /** The coefficient of the linear term in y */
  double _d;

  /** The constant offset */
  double _e;

  /** The Circle is a friend of the Surface class */
  friend class Surface;

  /** The Circle is a friend of the Plane class */
  friend class Plane;

public:
  Circle(const double x, const double y, const double radius, const int id=0);

  double getX0();
  double getY0();
  double getRadius();
  double getXMin();
  double getXMax();
  double getYMin();
  double getYMax();

  double evaluate(const Point* point) const;
  int intersection(Point* point, double angle, Point* points);

  std::string toString();
  void printString();
};


/**
 * @brief Finds the minimum distance to a Surface.
 * @details Finds the miniumum distance to a Surface from a Point with a
 *          given trajectory defined by an angle to this Surface. If the
 *          trajectory will not intersect the Surface, returns INFINITY.
 * @param point a pointer to the Point of interest
 * @param angle the angle defining the trajectory in radians
 * @param intersection a pointer to a Point for storing the intersection
 * @return the minimum distance to the Surface
 */
inline double Surface::getMinDistance(Point* point, double angle,
                                      Point* intersection) {

  /* Point array for intersections with this Surface */
  Point intersections[2];

  /* Find the intersection Point(s) */
  int num_inters = this->intersection(point, angle, intersections);
  double distance = INFINITY;

  /* If there is one intersection Point */
  if (num_inters == 1) {
    distance = intersections[0].distanceToPoint(point);
    intersection->setX(intersections[0].getX());
    intersection->setY(intersections[0].getY());
  }

  /* If there are two intersection Points */
  else if (num_inters == 2) {
    double dist1 = intersections[0].distanceToPoint(point);
    double dist2 = intersections[1].distanceToPoint(point);

    /* Determine which intersection Point is nearest */
    if (dist1 < dist2) {
      distance = dist1;
      intersection->setX(intersections[0].getX());
      intersection->setY(intersections[0].getY());
    }
    else {
      distance = dist2;
      intersection->setX(intersections[1].getX());
      intersection->setY(intersections[1].getY());
    }
  }

  return distance;
}


/**
 * @brief Evaluate a Point using the Plane's quadratic Surface equation.
 * @param point a pointer to the Point of interest
 * @return the value of Point in the Plane's quadratic equation
 */
inline double Plane::evaluate(const Point* point) const {
  double x = point->getX();
  double y = point->getY();

  //TODO: does not support ZPlanes
  return (_a * x + _b * y + _c);
}


/**
 * @brief Return the radius of the Circle.
 * @return the radius of the Circle
 */
inline double Circle::getRadius() {
  return this->_radius;
}


/**
 * @brief Evaluate a Point using the Circle's quadratic Surface equation.
 * @param point a pointer to the Point of interest
 * @return the value of Point in the equation
 */
inline double Circle::evaluate(const Point* point) const {
  double x = point->getX();
  double y = point->getY();
  return (_a * x * x + _b * y * y + _c * x + _d * y + _e);
}


#endif /* SURFACE_H_ */
