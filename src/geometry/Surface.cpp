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

#include <cmath>
#include "geometry/Surface.h"

using namespace std;

int Surface::_n = 0;

static int auto_id = 10000;

/**
 * @brief Returns an auto-generated unique surface ID.
 * @details This method is intended as a utility mehtod for user's writing
 *          OpenMOC input files. The method makes use of a static surface
 *          ID which is incremented each time the method is called to enable
 *          unique generation of monotonically increasing IDs. The method's
 *          first ID begins at 10000. Hence, user-defined surface IDs greater
 *          than or equal to 10000 are prohibited.
 */
int surf_id() {
  int id = auto_id;
  auto_id++;
  return id;
}


/**
 * @brief Constructor assigns unique ID and user-defined ID for a Surface.
 * @details Assigns a default boundary condition for this Surface to
 *          BOUNDARY_NONE.
 * @param id an optional user-defined Surface ID
 */
Surface::Surface(const int id){

  /* If the user did not define an optional ID, create one */
  if (id == 0)
    _id = surf_id();
  else if (id >= surf_id())
    log_printf(ERROR_LOG, "Unable to set the ID of a surface to %d since surface "
               "IDs greater than 10000 are probibited by OpenMOC.", id);
  /* Use the user-defined ID */
  else
    _id = id;

  _uid = _n;
  _n++;
  _boundary_type = BOUNDARY_NONE;
}


/**
 * @brief Destructor.
 */
Surface::~Surface() { }


/**
 * @brief Return the Surface's unique ID.
 * @return the Surface's unique ID
 */
int Surface::getUid() const {
  return _uid;
}


/**
 * @brief Return the Surface's user-defined ID.
 * @return the Surface's user-defined ID
 */
int Surface::getId() const {
  return _id;
}


/**
 * @brief Return the type of Surface (ie, XPLANE, CIRCLE, etc).
 * @return the Surface type
 */
surfaceType Surface::getSurfaceType() {
  return _surface_type;
}


/**
 * @brief Returns the type of boundary conditions for this Surface (REFLECTIVE,
 *        VACUUM or BOUNDARY_NONE)
 * @return the type of boundary condition type for this Surface
 */
boundaryType Surface::getBoundaryType(){
  return _boundary_type;
}


/**
 * @brief Sets the boundary condition type (ie, VACUUM or REFLECTIVE) for this
 *        Surface.
 * @param boundary_type the boundary condition type for this Surface
 */
void Surface::setBoundaryType(boundaryType boundary_type) {
  _boundary_type = boundary_type;
}


/**
 * @brief Return true or false if a Point is on or off of a Surface.
 * @param point pointer to the Point of interest
 * @return on (true) or off (false) the Surface
 */
bool Surface::isPointOnSurface(Point* point) {

  /* Uses a threshold to determine whether the point is on the Surface */
  if (abs(evaluate(point)) < ON_SURFACE_THRESH)
    return true;
  else
    return false;
}


/**
 * @brief Return true or false if a LocalCoord is on or off of a Surface.
 * @param coord pointer to the LocalCoord of interest
 * @return on (true) or off (false) the Surface
 */
bool Surface::isCoordOnSurface(LocalCoords* coord) {
  return isPointOnSurface(coord->getPoint());
}


/**
 * @brief Constructor.
 * @param id the Surface ID
 * @param A the first coefficient in \f$ A * x + B * y + C = 0 \f$
 * @param B the second coefficient in \f$ A * x + B * y + C = 0 \f$
 * @param C the third coefficient in \f$ A * x + B * y + C = 0 \f$
 */
Plane::Plane(const double A, const double B,
             const double C, const int id):
  Surface(id) {

  _surface_type = PLANE;
  _a = A;
  _b = B;
  _c = C;
}


/**
 * @brief Returns the minimum x value of -INFINITY on this Surface.
 * @return the minimum x value of -INFINITY
 */
double Plane::getXMin(){
  log_printf(ERROR_LOG, "Plane::getXMin() not yet implemented");
  return -std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the maximum x value of INFINITY on this Surface.
 * @return the maximum x value of INFINITY
 */
double Plane::getXMax(){
  log_printf(ERROR_LOG, "Plane::getXMax() not yet implemented");
  return std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the minimum y value of -INFINITY on this Surface.
 * @return the minimum y value of -INFINITY
 */
double Plane::getYMin(){
  log_printf(ERROR_LOG, "Plane::getYMin() not yet implemented");
  return -std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the maximum y value of INFINITY on this Surface.
 * @return the maximum y value of INFINITY
 */
double Plane::getYMax(){
  log_printf(ERROR_LOG, "Plane::getYMax() not yet implemented");
  return std::numeric_limits<double>::infinity();
}


/**
* @brief Finds the intersection Point with this Plane from a given Point and
*        trajectory defined by an angle.
* @param point pointer to the Point of interest
* @param angle the angle defining the trajectory in radians
* @param points pointer to a Point to store the intersection Point
* @return the number of intersection Points (0 or 1)
*/
inline int Plane::intersection(Point* point, double angle, Point* points) {

  double x0 = point->getX();
  double y0 = point->getY();

  int num = 0;                /* number of intersections */
  double xcurr, ycurr;        /* coordinates of current intersection point */

  /* The track is vertical */
  if ((fabs(angle - (M_PI / 2))) < 1.0e-10) {

    /* The plane is also vertical => no intersections */
    if (_b == 0)
      return 0;

    /* The plane is not vertical */
    else {
      xcurr = x0;
      ycurr = (-_a * x0 - _c) / _b;
      points->setCoords(xcurr, ycurr);

      /* Check that point is in same direction as angle */
      if (angle < M_PI && ycurr > y0)
        num++;
      else if (angle > M_PI && ycurr < y0)
        num++;
      return num;
    }
  }

  /* If the track isn't vertical */
  else {
    double m = sin(angle) / cos(angle);

    /* The plane and track are parallel, no intersections */
    if (fabs(-_a/_b - m) < 1e-11 && _b != 0)
      return 0;

    else {
      xcurr = -(_b * (y0 - m * x0) + _c) / (_a + _b * m);
      ycurr = y0 + m * (xcurr - x0);
      points->setCoords(xcurr, ycurr);

      if (angle < M_PI && ycurr > y0)
        num++;
      else if (angle > M_PI && ycurr < y0)
        num++;

      return num;
    }
  }
}


/**
 * @brief Converts this Plane's attributes to a character array.
 * @details The character array returned conatins the type of Plane (ie,
 *          PLANE) and the A, B, and C coefficients in the
 *          quadratic Surface equation.
 * @return a character array of this Plane's attributes
 */
std::string Plane::toString() {
  std::stringstream string;

  string << "Surface id = " << _id << ", type = PLANE " << ", A = "
         << _a << ", B = " << _b << ", C = " << _c;

  return string.str();
}


/**
 * @brief Prints a string representation of all of the Plane's objects to
 *        the console.
 */
void Plane::printString() {
  log_printf(RESULT_LOG, toString().c_str());
}


/**
 * @brief Constructor for a Plane perpendicular to the x-axis.
 * @param id the user-defined Surface id
 * @param x the location of the Plane along the x-axis
 */
XPlane::XPlane(const double x, const int id):
  Plane(0, 1, -x, id) {

  _surface_type = XPLANE;
  _x = x;
}


/**
 * @brief Set the location of this XPlane on the x-axis.
 * @param x the location of the XPlane on the x-axis
 */
void XPlane::setX(const double x) {
  _x = x;
}


/**
 * @brief Returns the location of the XPlane on the x-axis.
 * @return the location of the XPlane on the x-axis
 */
double XPlane::getX() {
  return _x;
}


/**
 * @brief Returns the minimum x value on the XPlane.
 * @return the minimum x value
 */
double XPlane::getXMin(){
  return _x;
}


/**
 * @brief Returns the maximum x value on the XPlane.
 * @return the maximum x value
 */
double XPlane::getXMax(){
  return _x;
}


/**
 * @brief Returns the minimum y value of -INFINITY on the XPlane.
 * @return the minimum y value of -INFINITY
 */
double XPlane::getYMin(){
  return -std::numeric_limits<double>::infinity();
}


/**
 * Returns the maximum y value of INFINITY on this XPlane.
 * @return the maximum y value of INFINITY
 */
double XPlane::getYMax(){
  return std::numeric_limits<double>::infinity();
}


/**
 * @brief Converts this XPlane's attributes to a character array.
 * @details The character array returned conatins the type of Plane (ie,
 *          XPLANE) and the A, B, and C coefficients in the
 *          quadratic Surface equation and the location of the Plane on
 *          the x-axis.
 * @return a character array of this XPlane's attributes
 */
std::string XPlane::toString() {

  std::stringstream string;

  string << "Surface id = " << _id << ", type = XPLANE " << ", A = "
         << _a << ", B = " << _b << ", C = " << _c << ", x = " << _x;

  return string.str();
}


/**
 * @brief Constructor for a Plane perpendicular to the y-axis.
 * @param id the surface id
 * @param y the location of the Plane along the y-axis
 */
YPlane::YPlane(const double y, const int id):
  Plane(1, 0, -y, id) {

  _surface_type = YPLANE;
  _y = y;
}


/**
 * @brief Set the location of this YPlane on the y-axis.
 * @param y the location of the YPlane on the y-axis
 */
void YPlane::setY(const double y) {
  _y = y;
}


/**
 * @brief Returns the location of the YPlane on the y-axis.
 * @return the location of the YPlane on the y-axis
 */
double YPlane::getY() {
  return _y;
}

/**
 * @brief Returns the minimum x value of -INFINITY on this YPlane.
 * @return the minimum x value of -INFINITY
 */
double YPlane::getXMin(){
  return -std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the maximum x value of INFINITY on this YPlane.
 * @return the maximum x value of INFINITY
 */
double YPlane::getXMax(){
  return std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the minimum y value on this YPlane.
 * @return the minimum y value
 */
double YPlane::getYMin(){
  return _y;
}


/**
 * @brief Returns the maximum y value on this YPlane.
 * @return the maximum y value
 */
double YPlane::getYMax(){
  return _y;
}


/**
 * @brief Converts this yplane's attributes to a character array
 * @details The character array returned conatins the type of Plane (ie,
 *          YPLANE) and the A, B, and C coefficients in the quadratic
 *          Surface equation and the location of the Plane on the y-axis.
 * @return a character array of this YPlane's attributes
 */
std::string YPlane::toString() {

  std::stringstream string;

  string << "Surface id = " << _id << ", type = YPLANE " << ", A = "
         << _a << ", B = " << _b << ", C = " << _c << ", y = " << _y;

  return string.str();
}


/**
 * @brief Prints a string representation of all of the YPlane's objects to
 *        the console.
 */
void YPlane::printString() {
  log_printf(RESULT_LOG, toString().c_str());
}


/**
 * @brief Constructor for a Plane perpendicular to the z-axis.
 * @param id the surface ID
 * @param z the location of the Plane along the z-axis
 */
ZPlane::ZPlane(const double z, const int id):
  Plane(0, 0, -z, id) {

  _surface_type = ZPLANE;
  _z = z;
}


/**
 * @brief Set the location of this ZPlane on the z-axis.
 * @param z the location of the ZPlane on the z-axis
 */
void ZPlane::setZ(const double z) {
  _z = z;
}


/**
 * @brief Returns the location of the ZPlane on the z-axis.
 * @return the location of the ZPlane on the z-axis
 */
double ZPlane::getZ() {
  return _z;
}


/**
 * @brief Returns the minimum x value of -INFINITY on this ZPlane.
 * @return the minimum x value of -INFINITY
 */
double ZPlane::getXMin(){
  return -std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the maximum x value of INFINITY on this ZPlane.
 * @return the maximum x value of INFINITY
 */
double ZPlane::getXMax(){
  return std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the minimum x value of -INFINITY on this ZPlane.
 * @return the minimum y value of -INFINITY
 */
double ZPlane::getYMin(){
  return -std::numeric_limits<double>::infinity();
}


/**
 * @brief Returns the maximum y value on this ZPlane.
 * @return the maximum y value
 */
double ZPlane::getYMax(){
  return std::numeric_limits<double>::infinity();
}

/**
 * @brief Converts this ZPlane's attributes to a character array.
 * @details The character array returned conatins the type of Plane (ie,
 *          ZPLANE) and the A, B, and C coefficients in the
 *          quadratic Surface equation and the location of the Plane along
 *          the z-axis.
 * @return a character array of this ZPlane's attributes
 */
std::string ZPlane::toString() {
  std::stringstream string;

  string << "Surface id = " << _id << ", type = ZPLANE " << ", A = "
         << _a << ", B = " << _b << ", C = " << _c << ", z = " << _z;

  return string.str();
}


/**
 * @brief Prints a string representation of all of the ZPlane's objects to
 *        the console.
 */
void ZPlane::printString() {
  log_printf(RESULT_LOG, toString().c_str());
}


/**
 * @brief constructor.
 * @param id the surface ID
 * @param x the x-coordinte of the Circle center
 * @param y the y-coordinate of the Circle center
 * @param radius the radius of the Circle
 */
Circle::Circle(const double x, const double y,
               const double radius, const int id):
  Surface(id) {

  _surface_type = CIRCLE;
  _a = 1.;
  _b = 1.;
  _c = -2.*x;
  _d = -2.*y;
  _e = x*x + y*y - radius*radius;
  _radius = radius;
  _center.setX(x);
  _center.setY(y);
}


/**
 * @brief Return the x-coordinate of the Circle's center Point.
 * @return the x-coordinate of the Circle center
 */
double Circle::getX0() {
  return _center.getX();
}


/**
 * @brief Return the y-coordinate of the Circle's center Point.
 * @return the y-coordinate of the Circle center
 */
double Circle::getY0() {
  return _center.getY();
}


/**
 * @brief Finds the intersection Point with this circle from a given Point and
 *        trajectory defined by an angle (0, 1, or 2 points).
 * @param point pointer to the Point of interest
 * @param angle the angle defining the trajectory in radians
 * @param points pointer to a an array of Points to store intersection Points
 * @return the number of intersection Points (0 or 1)
 */
int Circle::intersection(Point* point, double angle, Point* points) {

  double x0 = point->getX();
  double y0 = point->getY();
  double xcurr, ycurr;
  int num = 0;                        /* Number of intersection Points */
  double a, b, c, q, discr;

  /* If the track is vertical */
  if ((fabs(angle - (M_PI / 2))) < 1.0e-10) {

    /* Solve for where the line x = x0 and the Surface F(x,y) intersect
     * Find the y where F(x0, y) = 0
     * Substitute x0 into F(x,y) and rearrange to put in
     * the form of the quadratic formula: ay^2 + by + c = 0 */
    a = _b * _b;
    b = _d;
    c = _a * x0 * x0 + _c * x0 + _e;

    discr = b*b - 4*a*c;

    /* There are no intersections */
    if (discr < 0)
      return 0;

    /* There is one intersection (ie on the Surface) */
    else if (discr == 0) {
      xcurr = x0;
      ycurr = -b / (2*a);
      points[num].setCoords(xcurr, ycurr);
      if (angle < M_PI && ycurr > y0)
        num++;
      else if (angle > M_PI && ycurr < y0)
        num++;
      return num;
    }

    /* There are two intersections */
    else {
      xcurr = x0;
      ycurr = (-b + sqrt(discr)) / (2 * a);
      points[num].setCoords(xcurr, ycurr);
      if (angle < M_PI && ycurr > y0)
        num++;
      else if (angle > M_PI && ycurr < y0)
        num++;

      xcurr = x0;
      ycurr = (-b - sqrt(discr)) / (2 * a);
      points[num].setCoords(xcurr, ycurr);
      if (angle < M_PI && ycurr > y0)
        num++;
      else if (angle > M_PI && ycurr < y0)
        num++;
      return num;
    }
  }

  /* If the track isn't vertical */
  else {
    /* Solve for where the line y-y0 = m*(x-x0) and the Surface F(x,y)
     * intersect. Find the (x,y) where F(x, y0 + m*(x-x0)) = 0
     * Substitute the point-slope formula for y into F(x,y) and
     * rearrange to put in the form of the quadratic formula:
     * ax^2 + bx + c = 0
     */
    double m = sin(angle) / cos(angle);
    q = y0 - m * x0;
    a = _a + _b * _b * m * m;
    b = 2 * _b * m * q + _c + _d * m;
    c = _b * q * q + _d * q + _e;

    discr = b*b - 4*a*c;

    /* There are no intersections */
    if (discr < 0)
      return 0;

    /* There is one intersection (ie on the Surface) */
    else if (discr == 0) {
      xcurr = -b / (2*a);
      ycurr = y0 + m * (points[0].getX() - x0);
      points[num].setCoords(xcurr, ycurr);
      if (angle < M_PI && ycurr > y0)
        num++;
      else if (angle > M_PI && ycurr < y0)
        num++;
      return num;
    }

    /* There are two intersections */
    else {
      xcurr = (-b + sqrt(discr)) / (2*a);
      ycurr = y0 + m * (xcurr - x0);
      points[num].setCoords(xcurr, ycurr);
      if (angle < M_PI && ycurr > y0) {
        num++;
      }
      else if (angle > M_PI && ycurr < y0) {
        num++;
      }

      xcurr = (-b - sqrt(discr)) / (2*a);
      ycurr = y0 + m * (xcurr - x0);
      points[num].setCoords(xcurr, ycurr);
      if (angle < M_PI && ycurr > y0) {
        num++;
      }
      else if (angle > M_PI && ycurr < y0) {
        num++;
      }

      return num;
    }
  }
}


/**
 * @brief Converts this Circle's attributes to a character array.
 * @details The character array returned conatins the type of Plane (ie,
 *          CIRCLE) and the A, B, C, D and E coefficients in the
 *          quadratic Surface equation.
 * @return a character array of this Circle's attributes
 */
std::string Circle::toString() {
  std::stringstream string;

  string << "Surface id = " << _id << ", type = CIRCLE " << ", A = "
         << _a << ", B = " << _b << ", C = " << _c << ", D = " << _d
         << ", E = " << _e << ", x0 = " << _center.getX() << ", y0 = "
         << _center.getY() << ", radius = " << _radius;

    return string.str();
}


/**
 * @brief Prints a string representation of all of the Circle's attributes to
 *        the console.
 */
void Circle::printString() {
  log_printf(RESULT_LOG, toString().c_str());
}


/**
 * @brief Returns the minimum x value on this Circle.
 * @return the minimum y value
 */
double Circle::getXMin(){
  return _center.getX() - _radius;
}


/**
 * @brief Returns the maximum x value on this Circle.
 * @return the maximum x value
 */
double Circle::getXMax(){
  return _center.getX() + _radius;
}

/**
 * @brief Returns the minimum y value on this Circle.
 * @return the minimum y value
 */
double Circle::getYMin(){
  return _center.getY() - _radius;
}


/**
 * @brief Returns ths maximum y value on this Circle.
 * @return the maximum y value
 */
double Circle::getYMax(){
  return _center.getY() + _radius;
}
