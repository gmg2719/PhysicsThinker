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

#ifndef _TRACKGENERATOR_H_
#define _TRACKGENERATOR_H_      1

#include "Basic.h"

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file TrackGenerator.h
 * @brief The TrackGenerator class.
 * @date January 23, 2012
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifdef __cplusplus
  #include <iostream>
  #include <fstream>
  #include <sstream>
  #include <unistd.h>
#ifdef _OPENMP
  #include <omp.h>
#endif
  #include "Track.h"
  #include "Geometry.h"
#endif


/**
 * @class TrackGenerator TrackGenerator.h "src/TrackGenerator.h"
 * @brief The TrackGenerator is dedicated to generating and storing Tracks
 *        which cyclically wrap across the Geometry.
 * @details The TrackGenerator creates Track and initializes boundary
 *          conditions (vacuum or reflective) for each Track.
 */
class TrackGenerator {

private:

  /** Number of azimuthal angles in \f$ [0, \pi] \f$ */
  int _num_azim;

  /** The track spacing (cm) */
  double _spacing;

  /** An integer array of the number of Tracks for each azimuthal angle */
  int* _num_tracks;

  /** The total number of Tracks for all azimuthal angles */
  int _tot_num_tracks;

  /** An integer array of the number of segments per Track  */
  int* _num_segments;

  /** The total number of segments for all Tracks */
  int _tot_num_segments;

  /** An integer array of the number of Tracks starting on the x-axis for each
   *  azimuthal angle */
  int* _num_x;

  /** An integer array of the number of Tracks starting on the y-axis for each
   *  azimuthal angle */
  int* _num_y;

  /** An array of the azimuthal angle quadrature weights */
  FP_PRECISION* _azim_weights;

  /** A 2D ragged array of Tracks */
  Track** _tracks;

  /** Pointer to the Geometry */
  Geometry* _geometry;

  /** Boolean for whether to use Track input file (true) or not (false) */
  bool _use_input_file;

  /** Filename for the *.tracks input / output file */
  std::string _tracks_filename;

  /** Boolean whether the Tracks have been generated (true) or not (false) */
  bool _contains_tracks;

  void computeEndPoint(Point* start, Point* end,  const double phi,
                       const double width, const double height);

  void initializeTrackFileDirectory();
  void initializeTracks();
  void recalibrateTracksToOrigin();
  void initializeBoundaryConditions();
  void segmentize();
  void dumpTracksToFile();
  bool readTracksFromFile();

public:
  TrackGenerator(Geometry* geometry, int num_azim, double spacing);
  virtual ~TrackGenerator();

  int getNumAzim();
  double getTrackSpacing();
  Geometry* getGeometry();
  int getNumTracks();
  int* getNumTracksArray();
  int getNumSegments();
  int* getNumSegmentsArray();
  Track** getTracks();
  FP_PRECISION* getAzimWeights();

  void setNumAzim(int num_azim);
  void setTrackSpacing(double spacing);
  void setGeometry(Geometry* geometry);

  bool containsTracks();
  void retrieveTrackCoords(double* coords, int num_tracks);
  void retrieveSegmentCoords(double* coords, int num_segments);

  void generateTracks();
};

#endif /* TRACKGENERATOR_H_ */
