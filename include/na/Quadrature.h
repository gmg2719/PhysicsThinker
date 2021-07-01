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

#ifndef QUADRATURE_H_
#define QUADRATURE_H_

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//
/**
 * @file Quadrature.h
 * @brief The Quadrature class.
 * @date January 20, 2012
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
*/

typedef double FP_PRECISION;

#ifdef __cplusplus
  #include <sstream>
  #include "common/log.h"
#endif

/**
 * @enum quadratureType
 * @brief The type of polar quadrature.
 */
enum quadratureType {
  /** The Leonard type polar quadrature */
  LEONARD,

  /** The Tabuchi-Yamamoto polar quadrature */
  TABUCHI
};


/**
 * @class Quadrature Quadrature.h "src/Quadrature.h"
 * @brief Stores values for a variety of polar quadratures which may be used.
 */
class Quadrature {

private:
  /** The type of polar quadrature (LEONARD or TABUCHI) */
  quadratureType _type;

  /** The number of polar angles */
  int _num_polar_angles;

  /** An array of the sine of the polar angles from the quadrature set */
  FP_PRECISION* _sinthetas;

  /** An array of the weights for the polar angles from the quadrature set */
  FP_PRECISION* _weights;

  /** An array of the sine multipled by the weight for the polar angles from
   *  the quadrature set */
  FP_PRECISION* _multiples;

public:

  Quadrature(quadratureType type=TABUCHI, int num_polar_angles=3);
  virtual ~Quadrature();

  int getNumPolarAngles() const;
  quadratureType getType() const;
  FP_PRECISION getSinTheta(const int n) const;
  FP_PRECISION getWeight(const int n) const;
  FP_PRECISION getMultiple(const int n) const;
  FP_PRECISION* getSinThetas();
  FP_PRECISION* getWeights();
  FP_PRECISION* getMultiples();
  std::string toString();
};


#endif /* QUADRATURE_H_ */
