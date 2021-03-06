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
#include "geometry/TrackGenerator.h"

using namespace std;

/**
 * @brief Constructor for the TrackGenerator assigns default values.
 * @param geometry a pointer to a Geometry object
 * @param num_azim number of azimuthal angles in \f$ [0, 2\pi] \f$
 * @param spacing track spacing (cm)
 */
TrackGenerator::TrackGenerator(Geometry* geometry, const int num_azim,
                               const double spacing) {

  _geometry = geometry;
  setNumAzim(num_azim);
  setTrackSpacing(spacing);
  _tot_num_tracks = 0;
  _tot_num_segments = 0;
  _num_segments = NULL;
  _contains_tracks = false;
  _use_input_file = false;
  _tracks_filename = "";
}


/**
 * @brief Destructor frees memory for all Tracks.
 */
TrackGenerator::~TrackGenerator() {

  /* Deletes Tracks arrays if Tracks have been generated */
  if (_contains_tracks) {
    delete [] _num_tracks;
    delete [] _num_segments;
    delete [] _num_x;
    delete [] _num_y;
    delete [] _azim_weights;

    for (int i = 0; i < _num_azim; i++)
      delete [] _tracks[i];

    delete [] _tracks;
  }
}


/**
 * @brief Return the number of azimuthal angles in \f$ [0, 2\pi] \f$
 * @return the number of azimuthal angles in \f$ 2\pi \f$
 */
int TrackGenerator::getNumAzim() {
  return _num_azim * 2.0;
}


/**
 * @brief Return the track spacing (cm).
 * @details This will return the user-specified track spacing and NOT the
 *          effective track spacing which is computed and used to generate
 *          cyclic tracks.
 * @return the track spacing (cm)
 */
double TrackGenerator::getTrackSpacing() {
  return _spacing;
}


/**
 * @brief Return the Geometry for this TrackGenerator if one has been set.
 * @return a pointer to the Geometry
 */
Geometry* TrackGenerator::getGeometry() {
  if (_geometry == NULL)
    log_printf(ERROR_LOG, "Unable to return the TrackGenerator's Geometry "
               "since it has not yet been set");

  return _geometry;
}


/**
 * @brief Return the total number of Tracks across the Geometry.
 * @return the total number of Tracks
 */
int TrackGenerator::getNumTracks() {

  if (!_contains_tracks)
    log_printf(ERROR_LOG, "Unable to return the total number of Tracks since "
               "Tracks have not yet been generated.");

  return _tot_num_tracks;
}

/**
 * @brief Return an array of the number of Tracks for each azimuthal angle.
 * @return array with the number of Tracks
 */
int* TrackGenerator::getNumTracksArray() {
  if (!_contains_tracks)
    log_printf(ERROR_LOG, "Unable to return the array of the number of Tracks per "
               "azimuthal angle since Tracks have not yet been generated.");

  return _num_tracks;
}


/**
 * @brief Return the total number of Track segments across the Geometry.
 * @return the total number of Track segments
 */
int TrackGenerator::getNumSegments() {

  if (!_contains_tracks)
    log_printf(ERROR_LOG, "Unable to return the total number of segments since "
               "Tracks have not yet been generated.");

  return _tot_num_segments;
}


/**
 * @brief Return an array of the number of segments per Track.
 * @return array with the number of segments per Track
 */
int* TrackGenerator::getNumSegmentsArray() {
  if (!_contains_tracks)
    log_printf(ERROR_LOG, "Unable to return the array of the number of segments "
               "per Track since Tracks have not yet been generated.");

  return _num_segments;
}


/**
 * @brief Returns a 2D jagged array of the Tracks.
 * @details The first index into the array is the azimuthal angle and the
 *          second index is the Track number for a given azimuthal angle.
 * @return the 2D jagged array of Tracks
 */
Track **TrackGenerator::getTracks() {
  if (!_contains_tracks)
    log_printf(ERROR_LOG, "Unable to return the 2D ragged array of the Tracks "
               "since Tracks have not yet been generated.");

  return _tracks;
}


/**
 * @brief Return a pointer to the array of azimuthal angle quadrature weights.
 * @return the array of azimuthal angle quadrature weights
 */
FP_PRECISION* TrackGenerator::getAzimWeights() {
  if (!_contains_tracks)
    log_printf(ERROR_LOG, "Unable to return Track azimuthal angle quadrature "
               "weights since Tracks have not yet been generated.");

  return _azim_weights;
}


/**
 * @brief Returns whether or not the TrackGenerator contains Track that are
 *        for its current number of azimuthal angles, track spacing and
 *        geometry.
 * @return true if the TrackGenerator conatains Tracks; false otherwise
 */
bool TrackGenerator::containsTracks() {
  return _contains_tracks;
}


/**
 * @brief Fills an array with the x,y coordinates for each Track.
 * @details This class method is intended to be called by the OpenMOC
 *          Python "plotter" module as a utility to assist in plotting
 *          tracks. Although this method appears to require two arguments,
 *          in reality it only requires on due to SWIG and would be called
 *          from within Python as follows:
 *
 * @code
 *          num_tracks = track_generator.getNumTracks()
 *          coords = track_generator.retrieveTrackCoords(num_tracks*4)
 * @endcode
 *
 * @param coords an array of coords of length 4 times the number of Tracks
 * @param num_tracks the total number of Tracks
 */
void TrackGenerator::retrieveTrackCoords(double* coords, int num_tracks) {

  if (num_tracks != 4*getNumTracks())
    log_printf(ERROR_LOG, "Unable to retrieve the Track coordinates since the "
               "TrackGenerator contains %d Tracks with %d coordinates but an "
               "array of length %d was input",
               getNumTracks(), 4*getNumTracks(), num_tracks);

  /* Fill the array of coordinates with the Track start and end points */
  int counter = 0;
  for (int i=0; i < _num_azim; i++) {
    for (int j=0; j < _num_tracks[i]; j++) {
      coords[counter] = _tracks[i][j].getStart()->getX();
      coords[counter+1] = _tracks[i][j].getStart()->getY();
      coords[counter+2] = _tracks[i][j].getEnd()->getX();
      coords[counter+3] = _tracks[i][j].getEnd()->getY();
      counter += 4;
    }
  }

  return;
}


/**
 * @brief Fills an array with the x,y coordinates for each Track segment.
 * @details This class method is intended to be called by the OpenMOC
 *          Python "plotter" module as a utility to assist in plotting
 *          segments. Although this method appears to require two arguments,
 *          in reality it only requires one due to SWIG and would be called
 *          from within Python as follows:
 *
 * @code
 *          num_segments = track_generator.getNumSegments()
 *          coords = track_generator.retrieveSegmentCoords(num_segments*5)
 * @endcode
 *
 * @param coords an array of coords of length 5 times the number of segments
 * @param num_segments the total number of Track segments
 */
void TrackGenerator::retrieveSegmentCoords(double* coords, int num_segments) {

  if (num_segments != 5*getNumSegments())
    log_printf(ERROR_LOG, "Unable to retrieve the Track segment coordinates since "
               "the TrackGenerator contains %d segments with %d coordinates "
               "but an array of length %d was input",
               getNumSegments(), 5*getNumSegments(), num_segments);

  segment* curr_segment = NULL;
  double x0, x1, y0, y1;
  double phi;
  segment* segments;

  int counter = 0;

  /* Loop over Track segments and populate array with their FSR ID and *
   * start/end points */
  for (int i=0; i < _num_azim; i++) {
    for (int j=0; j < _num_tracks[i]; j++) {

      x0 = _tracks[i][j].getStart()->getX();
      y0 = _tracks[i][j].getStart()->getY();
      phi = _tracks[i][j].getPhi();

      segments = _tracks[i][j].getSegments();

      for (int s=0; s < _tracks[i][j].getNumSegments(); s++) {
        curr_segment = &segments[s];

        coords[counter] = curr_segment->_region_id;

        coords[counter+1] = x0;
        coords[counter+2] = y0;

        x1 = x0 + cos(phi) * curr_segment->_length;
        y1 = y0 + sin(phi) * curr_segment->_length;

        coords[counter+3] = x1;
        coords[counter+4] = y1;

        x0 = x1;
        y0 = y1;

        counter += 5;
      }
    }
  }

    return;
}


/**
 * @brief Set the number of azimuthal angles in \f$ [0, 2\pi] \f$.
 * @param num_azim the number of azimuthal angles in \f$ 2\pi \f$
 */
void TrackGenerator::setNumAzim(int num_azim) {

  if (num_azim < 0)
    log_printf(ERROR_LOG, "Unable to set a negative number of azimuthal angles "
               "%d for the TrackGenerator.", num_azim);

  if (num_azim % 4 != 0)
    log_printf(ERROR_LOG, "Unable to set the number of azimuthal angles to %d for "
               "the TrackGenerator since it is not a multiple of 4", num_azim);

  /* Subdivide out angles in [pi,2pi] */
  _num_azim = num_azim / 2.0;

  _contains_tracks = false;
  _use_input_file = false;
  _tracks_filename = "";
}


/**
 * @brief Set the suggested track spacing (cm).
 * @param spacing the suggested track spacing
 */
void TrackGenerator::setTrackSpacing(double spacing) {
  if (spacing < 0)
    log_printf(ERROR_LOG, "Unable to set a negative track spacing %f for the "
               "TrackGenerator.", spacing);

  _spacing = spacing;
  _tot_num_tracks = 0;
  _tot_num_segments = 0;
  _contains_tracks = false;
  _use_input_file = false;
  _tracks_filename = "";
}


/**
 * @brief Set a pointer to the Geometry to use for track generation.
 * @param geometry a pointer to the Geometry
 */
void TrackGenerator::setGeometry(Geometry* geometry) {
  _geometry = geometry;
  _tot_num_tracks = 0;
  _tot_num_segments = 0;
  _contains_tracks = false;
  _use_input_file = false;
  _tracks_filename = "";
}

/**
 * @brief Generates tracks for some number of azimuthal angles and track spacing
 * @details Computes the effective angles and track spacing. Computes the
 *          number of Tracks for each azimuthal angle, allocates memory for
 *          all Tracks at each angle and sets each Track's starting and ending
 *          Points, azimuthal angle, and azimuthal angle quadrature weight.
 */
void TrackGenerator::generateTracks() {

  if (_geometry == NULL)
    log_printf(ERROR_LOG, "Unable to generate Tracks since no Geometry "
               "has been set for the TrackGenerator");

  /* Deletes Tracks arrays if Tracks have been generated */
  if (_contains_tracks) {
    delete [] _num_tracks;
    delete [] _num_segments;
    delete [] _num_x;
    delete [] _num_y;
    delete [] _azim_weights;

    for (int i = 0; i < _num_azim; i++)
      delete [] _tracks[i];

    delete [] _tracks;
  }

  initializeTrackFileDirectory();

  /* If not Tracks input file exists, generate Tracks */
  if (_use_input_file == false) {

    /* Allocate memory for the Tracks */
    try {
      _num_tracks = new int[_num_azim];
      _num_x = new int[_num_azim];
      _num_y = new int[_num_azim];
      _azim_weights = new FP_PRECISION[_num_azim];
      _tracks = new Track*[_num_azim];
    }
    catch (std::exception &e) {
      log_printf(ERROR_LOG, "Unable to allocate memory for TrackGenerator. "
                 "Backtrace:\n%s", e.what());
    }

    /* Check to make sure that height, width of the Geometry are nonzero */
    if (_geometry->getHeight() <= 0 || _geometry->getHeight() <= 0)
      log_printf(ERROR_LOG, "The total height and width of the Geometry must be"
                 "must be nonzero for Track generation. Create a CellFill which"
                 "is filled by the entire geometry and bounded by XPlanes"
                 "and YPlanes to enable the Geometry to determine the total"
                 "width and height of the model.");

    /* Generate Tracks, perform ray tracing across the geometry, and store
     * the data to a Track file */
    try {
      initializeTracks();
      recalibrateTracksToOrigin();
      segmentize();
      dumpTracksToFile();
    }
    catch (std::exception &e) {
      log_printf(ERROR_LOG, "Unable to allocate memory needed to generate "
                 "Tracks. Backtrace:\n%s", e.what());
    }
  }

  initializeBoundaryConditions();
  return;
}


/**
 * @brief This method creates a directory to store Track files, and reads
 *        in ray tracing data for Tracks and segments from a Track file
 *        if one exists.
 * @details This method is called by the TrackGenerator::generateTracks()
 *          class method. If a Track file exists for this Geometry, number
 *          of azimuthal angles, and track spacing, then this method will
 *          import the ray tracing Track and segment data to fill the
 *          appropriate data structures.
 */
void TrackGenerator::initializeTrackFileDirectory() {

  std::stringstream directory;
  struct stat buffer;
  std::stringstream test_filename;

  /** Create directory to store Track files with pre-generated ray tracing data
   *  if the directory does not yet exist */

  directory << get_output_directory() << "/tracks";
  struct stat st;
  if (!stat(directory.str().c_str(), &st) == 0)
    mkdir(directory.str().c_str(), S_IRWXU);

  if (_geometry->getMesh()->getCmfdOn()){
    test_filename << directory.str() << "/tracks_"
                  <<  _num_azim*2.0 << "_angles_"
                  << _spacing << "_cm_spacing_cmfd_"
                  << _geometry->getMesh()->getMeshLevel() << ".data";
    }
  else{
    test_filename << directory.str() << "/tracks_"
                  <<  _num_azim*2.0 << "_angles_"
                  << _spacing << "_cm_spacing.data";
  }

  _tracks_filename = test_filename.str();

  /* Check to see if a Track file exists for this geometry, number of azimuthal
   * angles, and track spacing, and if so, import the ray tracing data */
  if (!stat(_tracks_filename.c_str(), &buffer)) {
    if (readTracksFromFile()) {
      _use_input_file = true;
      _contains_tracks = true;
    }
  }
}


/**
 * @brief Initializes Track azimuthal angles, start and end Points.
 * @details This method computes the azimuthal angles and effective track
 *          spacing to use to guarantee cyclic Track wrapping. Based on the
 *          angles and spacing, the number of Tracks per angle and the start
 *          and end Points for each Track are computed.
 */
void TrackGenerator::initializeTracks() {

  log_printf(INFO_LOG, "Computing azimuthal angles and track spacing...");

  /* Each element in arrays corresponds to an angle in phi_eff */
  /* Track spacing along x,y-axes, and perpendicular to each Track */
  double* dx_eff = new double[_num_azim];
  double* dy_eff = new double[_num_azim];
  double* d_eff = new double[_num_azim];

  /* Effective azimuthal angles with respect to positive x-axis */
  double* phi_eff = new double[_num_azim];

  double x1, x2;
  double iazim = _num_azim*2.0;
  double width = _geometry->getWidth();
  double height = _geometry->getHeight();

  /* Determine azimuthal angles and track spacing */
  for (int i = 0; i < _num_azim; i++) {

    /* A desired azimuthal angle for the user-specified number of
     * azimuthal angles */
    double phi = 2.0 * M_PI / iazim * (0.5 + i);

    /* The number of intersections with x,y-axes */
    _num_x[i] = (int) (fabs(width / _spacing * sin(phi))) + 1;
    _num_y[i] = (int) (fabs(height / _spacing * cos(phi))) + 1;

    /* Total number of Tracks */
    _num_tracks[i] = _num_x[i] + _num_y[i];

    /* Effective/actual angle (not the angle we desire, but close) */
    phi_eff[i] = atan((height * _num_x[i]) / (width * _num_y[i]));

    /* Fix angles in range(pi/2, pi) */
    if (phi > M_PI / 2)
      phi_eff[i] = M_PI - phi_eff[i];

    /* Effective Track spacing (not spacing we desire, but close) */
    dx_eff[i] = (width / _num_x[i]);
    dy_eff[i] = (height / _num_y[i]);
    d_eff[i] = (dx_eff[i] * sin(phi_eff[i]));
  }

  /* Compute azimuthal angle quadrature weights */
  for (int i = 0; i < _num_azim; i++) {

    if (i < _num_azim - 1)
      x1 = 0.5 * (phi_eff[i+1] - phi_eff[i]);
    else
      x1 = 2 * M_PI / 2.0 - phi_eff[i];

    if (i >= 1)
      x2 = 0.5 * (phi_eff[i] - phi_eff[i-1]);
    else
      x2 = phi_eff[i];

    /* Multiply weight by 2 because angles are in [0, Pi] */
    _azim_weights[i] = (x1 + x2) / (2 * M_PI) * d_eff[i] * 2;
  }

  log_printf(INFO_LOG, "Generating Track start and end points...");

  /* Compute Track starting and end points */
  for (int i = 0; i < _num_azim; i++) {

    /* Tracks for azimuthal angle i */
    _tracks[i] = new Track[_num_tracks[i]];

    /* Compute start points for Tracks starting on x-axis */
    for (int j = 0; j < _num_x[i]; j++)
      _tracks[i][j].getStart()->setCoords(dx_eff[i] * (0.5+j), 0);

    /* Compute start points for Tracks starting on y-axis */
    for (int j = 0; j < _num_y[i]; j++) {

      /* If Track points to the upper right */
      if (sin(phi_eff[i]) > 0 && cos(phi_eff[i]) > 0)
        _tracks[i][_num_x[i]+j].getStart()->setCoords(0,
                                     dy_eff[i] * (0.5 + j));

      /* If Track points to the upper left */
      else if (sin(phi_eff[i]) > 0 && cos(phi_eff[i]) < 0)
        _tracks[i][_num_x[i]+j].getStart()->setCoords(width,
                                     dy_eff[i] * (0.5 + j));
    }

    /* Compute the exit points for each Track */
    for (int j = 0; j < _num_tracks[i]; j++) {

      /* Set the Track's end point */
      Point* start = _tracks[i][j].getStart();
      Point* end = _tracks[i][j].getEnd();
      computeEndPoint(start, end, phi_eff[i], width, height);

      /* Set the Track's azimuthal angle */
      _tracks[i][j].setPhi(phi_eff[i]);
    }
  }

  delete [] dx_eff;
  delete [] dy_eff;
  delete [] d_eff;
  delete [] phi_eff;
}


/**
 * @brief Recalibrates Track start and end points to the origin of the Geometry.
 * @details The origin of the Geometry is designated at its center by
 *          convention, but for track initialization the origin is assumed to be
 *          at the bottom right corner for simplicity. This method corrects
 *          for this by re-assigning the start and end Point coordinates.
 */
void TrackGenerator::recalibrateTracksToOrigin() {

  int uid = 0;

  for (int i = 0; i < _num_azim; i++) {
    _tot_num_tracks += _num_tracks[i];

    for (int j = 0; j < _num_tracks[i]; j++) {

      _tracks[i][j].setUid(uid);
      uid++;

      double x0 = _tracks[i][j].getStart()->getX();
      double y0 = _tracks[i][j].getStart()->getY();
      double x1 = _tracks[i][j].getEnd()->getX();
      double y1 = _tracks[i][j].getEnd()->getY();
      double new_x0 = x0 - _geometry->getWidth()/2.0;
      double new_y0 = y0 - _geometry->getHeight()/2.0;
      double new_x1 = x1 - _geometry->getWidth()/2.0;
      double new_y1 = y1 - _geometry->getHeight()/2.0;
      double phi = _tracks[i][j].getPhi();

      _tracks[i][j].setValues(new_x0, new_y0, new_x1,new_y1, phi);
      _tracks[i][j].setAzimAngleIndex(i);
    }
  }
}


/**
 * @brief This helper method for TrackGenerator::generateTracks() finds the end
 *        Point of a Track with a defined start Point and an angle from x-axis.
 * @details This function does not return a value but instead saves the x/y
 *          coordinates of the end Point directly within the Track's end Point.
 * @param start pointer to the Track start Point
 * @param end pointer to a Point to store the end Point coordinates
 * @param phi the azimuthal angle
 * @param width the width of the Geometry (cm)
 * @param height the height of the Geometry (cm)
 */
void TrackGenerator::computeEndPoint(Point* start, Point* end,
                                     const double phi, const double width,
                                     const double height) {

  double m = sin(phi) / cos(phi);             /* slope */
  double yin = start->getY();                 /* y-coord */
  double xin = start->getX();                 /* x-coord */

  /* Allocate memory for the possible intersection points */
  Point *points = new Point[4];

  /* Determine all possible Points */
  points[0].setCoords(0, yin - m * xin);
  points[1].setCoords(width, yin + m * (width - xin));
  points[2].setCoords(xin - yin / m, 0);
  points[3].setCoords(xin - (yin - height) / m, height);

  /* For each of the possible intersection Points */
  for (int i = 0; i < 4; i++) {
    /* neglect the trivial Point (xin, yin) */
    if (points[i].getX() == xin && points[i].getY() == yin) { }

    /* The Point to return will be within the bounds of the cell */
    else if (points[i].getX() >= 0 && points[i].getX() <= width
             && points[i].getY() >= 0 && points[i].getY() <= height) {
      end->setCoords(points[i].getX(), points[i].getY());
    }
  }

    delete[] points;

    return;
}


/**
 * @brief Initializes boundary conditions for each Track.
 * @details Sets boundary conditions by setting the incoming and outgoing Tracks
 *          for each Track using a special indexing scheme into the 2D jagged
 *          array of Tracks.
 */
void TrackGenerator::initializeBoundaryConditions() {

  log_printf(INFO_LOG, "Initializing Track boundary conditions...");

  /* nxi = number of tracks starting on y-axis for angle i
   * nyi = number of tracks starting on y-axis for angle i
   * nti = total number of tracks for angle i */
  int nxi, nyi, nti;

  Track *curr;
  Track *refl;

  /* Loop over only half the angles since we will set the pointers for
   * connecting Tracks at the same time */
  for (int i = 0; i < floor(_num_azim / 2); i++) {
    nxi = _num_x[i];
    nyi = _num_y[i];
    nti = _num_tracks[i];
    curr = _tracks[i];
    refl = _tracks[_num_azim - i - 1];

    /* Loop over all of the Tracks for this angle */
    for (int j = 0; j < nti; j++) {

      /* More Tracks starting along x-axis than y-axis */
      if (nxi <= nyi) {

        /* Bottom to right hand side */
        if (j < nxi) {
          curr[j].setTrackIn(&refl[j]);
          curr[j].setTrackInI(_num_azim - i - 1);
          curr[j].setTrackInJ(j);

          refl[j].setTrackIn(&curr[j]);
          refl[j].setTrackInI(i);
          refl[j].setTrackInJ(j);

          curr[j].setReflIn(false);
          refl[j].setReflIn(false);

          if (_geometry->getBCBottom() == REFLECTIVE) {
            curr[j].setBCIn(1);
            refl[j].setBCIn(1);
          }
          else {
            curr[j].setBCIn(0);
            refl[j].setBCIn(0);
          }

          curr[j].setTrackOut(&refl[2 * nxi - 1 - j]);
          curr[j].setTrackOutI(_num_azim - i - 1);
          curr[j].setTrackOutJ(2 * nxi - 1 - j);

          refl[2 * nxi - 1 - j].setTrackIn(&curr[j]);
          refl[2 * nxi - 1 - j].setTrackInI(i);
          refl[2 * nxi - 1 - j].setTrackInJ(j);

          curr[j].setReflOut(false);
          refl[2 * nxi - 1 - j].setReflIn(true);

          if (_geometry->getBCRight() == REFLECTIVE) {
            curr[j].setBCOut(1);
            refl[2 * nxi - 1 - j].setBCIn(1);
          }
          else {
            curr[j].setBCOut(0);
            refl[2 * nxi - 1 - j].setBCIn(0);
          }
        }

        /* Left hand side to right hand side */
        else if (j < nyi) {
          curr[j].setTrackIn(&refl[j - nxi]);
          curr[j].setTrackInI(_num_azim - i - 1);
          curr[j].setTrackInJ(j - nxi);

          refl[j - nxi].setTrackOut(&curr[j]);
          refl[j - nxi].setTrackOutI(i);
          refl[j - nxi].setTrackOutJ(j);

          curr[j].setReflIn(true);
          refl[j - nxi].setReflOut(false);

          if (_geometry->getBCLeft() == REFLECTIVE) {
            curr[j].setBCIn(1);
            refl[j - nxi].setBCOut(1);
          }
          else {
            curr[j].setBCIn(0);
            refl[j - nxi].setBCOut(0);
          }

          curr[j].setTrackOut(&refl[j + nxi]);
          curr[j].setTrackOutI(_num_azim - i - 1);
          curr[j].setTrackOutJ(j + nxi);

          refl[j + nxi].setTrackIn(&curr[j]);
          refl[j + nxi].setTrackInI(i);
          refl[j + nxi].setTrackInJ(j);

          curr[j].setReflOut(false);
          refl[j + nxi].setReflIn(true);

          if (_geometry->getBCRight() == REFLECTIVE) {
            curr[j].setBCOut(1);
            refl[j + nxi].setBCIn(1);
          }
          else {
            curr[j].setBCOut(0);
            refl[j + nxi].setBCIn(0);
          }
        }

        /* Left hand side to top (j > ny) */
        else {
          curr[j].setTrackIn(&refl[j - nxi]);
          curr[j].setTrackInI(_num_azim - i - 1);
          curr[j].setTrackInJ(j - nxi);

          refl[j - nxi].setTrackOut(&curr[j]);
          refl[j - nxi].setTrackOutI(i);
          refl[j - nxi].setTrackOutJ(j);

          curr[j].setReflIn(true);
          refl[j - nxi].setReflOut(false);

          if (_geometry->getBCLeft() == REFLECTIVE) {
            curr[j].setBCIn(1);
            refl[j - nxi].setBCOut(1);
          }
          else {
            curr[j].setBCIn(0);
            refl[j - nxi].setBCOut(0);
          }

          curr[j].setTrackOut(&refl[2 * nti - nxi - j - 1]);
          curr[j].setTrackOutI(_num_azim - i - 1);
          curr[j].setTrackOutJ(2 * nti - nxi - j - 1);

          refl[2 * nti - nxi - j - 1].setTrackOut(&curr[j]);
          refl[2 * nti - nxi - j - 1].setTrackOutI(i);
          refl[2 * nti - nxi - j - 1].setTrackOutJ(j);

          curr[j].setReflOut(true);
          refl[2 * nti - nxi - j - 1].setReflOut(true);

          if (_geometry->getBCTop() == REFLECTIVE) {
            curr[j].setBCOut(1);
            refl[2 * nti - nxi - j - 1].setBCOut(1);
          }
          else {
            curr[j].setBCOut(0);
            refl[2 * nti - nxi - j - 1].setBCOut(0);
          }
        }
      }

      /* More Tracks starting on y-axis than on x-axis */
      else {

        /* Bottom to top */
        if (j < nxi - nyi) {
          curr[j].setTrackIn(&refl[j]);
          curr[j].setTrackInI(_num_azim - i - 1);
          curr[j].setTrackInJ(j);

          refl[j].setTrackIn(&curr[j]);
          refl[j].setTrackInI(i);
          refl[j].setTrackInJ(j);

          curr[j].setReflIn(false);
          refl[j].setReflIn(false);

          if (_geometry->getBCBottom() == REFLECTIVE) {
            curr[j].setBCIn(1);
            refl[j].setBCIn(1);
         }
          else {
            curr[j].setBCIn(0);
            refl[j].setBCIn(0);
          }

          curr[j].setTrackOut(&refl[nti - (nxi - nyi) + j]);
          curr[j].setTrackOutI(_num_azim - i - 1);
          curr[j].setTrackOutJ(nti - (nxi - nyi) + j);

          refl[nti - (nxi - nyi) + j].setTrackOut(&curr[j]);
          refl[nti - (nxi - nyi) + j].setTrackOutI(i);
          refl[nti - (nxi - nyi) + j].setTrackOutJ(j);

          curr[j].setReflOut(true);
          refl[nti - (nxi - nyi) + j].setReflOut(true);

          if (_geometry->getBCTop() == REFLECTIVE) {
            curr[j].setBCOut(1);
            refl[nti - (nxi - nyi) + j].setBCOut(1);
          }
          else {
            curr[j].setBCOut(0);
           refl[nti - (nxi - nyi) + j].setBCOut(0);
          }
        }

        /* Bottom to right hand side */
        else if (j < nxi) {
          curr[j].setTrackIn(&refl[j]);
          curr[j].setTrackInI(_num_azim - i - 1);
          curr[j].setTrackInJ(j);

          refl[j].setTrackIn(&curr[j]);
          refl[j].setTrackInI(i);
          refl[j].setTrackInJ(j);

          curr[j].setReflIn(false);
          refl[j].setReflIn(false);

          if (_geometry->getBCBottom() == REFLECTIVE) {
            curr[j].setBCIn(1);
            refl[j].setBCIn(1);
          }
          else {
            curr[j].setBCIn(0);
            refl[j].setBCIn(0);
          }

          curr[j].setTrackOut(&refl[nxi + (nxi - j) - 1]);
          curr[j].setTrackOutI(_num_azim - i - 1);
          curr[j].setTrackOutJ(nxi + (nxi - j) - 1);

          refl[nxi + (nxi - j) - 1].setTrackIn(&curr[j]);
          refl[nxi + (nxi - j) - 1].setTrackInI(i);
          refl[nxi + (nxi - j) - 1].setTrackInJ(j);

          curr[j].setReflOut(false);
          refl[nxi + (nxi - j) - 1].setReflIn(true);

          if (_geometry->getBCRight() == REFLECTIVE) {
            curr[j].setBCOut(1);
            refl[nxi + (nxi - j) - 1].setBCIn(1);
          }
          else {
            curr[j].setBCOut(0);
            refl[nxi + (nxi - j) - 1].setBCIn(0);
          }
        }

        /* Left-hand side to top (j > nx) */
        else {
          curr[j].setTrackIn(&refl[j - nxi]);
          curr[j].setTrackInI(_num_azim - i - 1);
          curr[j].setTrackInJ(j - nxi);

          refl[j - nxi].setTrackOut(&curr[j]);
          refl[j - nxi].setTrackOutI(i);
          refl[j - nxi].setTrackOutJ(j);

          curr[j].setReflIn(true);
          refl[j - nxi].setReflOut(false);

          if (_geometry->getBCLeft() == REFLECTIVE) {
            curr[j].setBCIn(1);
            refl[j - nxi].setBCOut(1);
          }
          else {
            curr[j].setBCIn(0);
            refl[j - nxi].setBCOut(0);
          }

          curr[j].setTrackOut(&refl[nyi + (nti - j) - 1]);
          curr[j].setTrackOutI(_num_azim - i - 1);
          curr[j].setTrackOutJ(nyi + (nti - j) - 1);

          refl[nyi + (nti - j) - 1].setTrackOut(&curr[j]);
          refl[nyi + (nti - j) - 1].setTrackOutI(i);
          refl[nyi + (nti - j) - 1].setTrackOutJ(j);

          curr[j].setReflOut(true);
          refl[nyi + (nti - j) - 1].setReflOut(true);

          if (_geometry->getBCTop() == REFLECTIVE) {
            curr[j].setBCOut(1);
            refl[nyi + (nti - j) - 1].setBCOut(1);
          }
          else {
            curr[j].setBCOut(0);
            refl[nyi + (nti - j) - 1].setBCOut(0);
          }
        }
      }
    }
  }

  return;
}


/**
 * @brief Generate segments for each Track across the Geometry.
 */
void TrackGenerator::segmentize() {

  log_printf(NORMAL_LOG, "Ray tracing for track segmentation...");

  Track* track;

  if (_num_segments != NULL)
    delete [] _num_segments;

  /* This section loops over all Track and segmentizes each one if the
   * Tracks were not read in from an input file */
  if (!_use_input_file) {

    /* Loop over all Tracks */
  #ifdef _OPENMP
    #pragma omp parallel for private(track)
  #endif
    for (int i=0; i < _num_azim; i++) {
      for (int j=0; j < _num_tracks[i]; j++){
        track = &_tracks[i][j];
        log_printf(DEBUG_LOG, "Segmenting Track %d/%d with i = %d, j = %d",
        track->getUid(), _tot_num_tracks, i, j);
        _geometry->segmentize(track);
      }
    }

    /* Compute the total number of segments in the simulation */
    _num_segments = new int[_tot_num_tracks];
    _tot_num_segments = 0;

    for (int i=0; i < _num_azim; i++) {
      for (int j=0; j < _num_tracks[i]; j++) {
        track = &_tracks[i][j];
        _num_segments[track->getUid()] = track->getNumSegments();
        _tot_num_segments += _num_segments[track->getUid()];
      }
    }
  }

    _contains_tracks = true;

    return;
}


/**
 * @brief Writes all Track and segment data to a "*.tracks" binary file.
 * @details Storing Tracks in a binary file saves time by eliminating ray
 *          tracing for Track segmentation in commonly simulated geometries.
 */
void TrackGenerator::dumpTracksToFile() {

  if (!_contains_tracks)
    log_printf(ERROR_LOG, "Unable to dump Tracks to a file since no Tracks have "
      "been generated for %d azimuthal angles and %f track spacing",
      _num_azim, _spacing);

  FILE* out;
  out = fopen(_tracks_filename.c_str(), "w");

  /* Get a string representation of the Geometry's attributes. This is used to
   * check whether or not ray tracing has been performed for this Geometry */
  std::string geometry_to_string = _geometry->toString();
  int string_length = geometry_to_string.length();

  /* Write geometry metadata to the Track file */
  fwrite(&string_length, sizeof(int), 1, out);
  fwrite(geometry_to_string.c_str(), sizeof(char)*string_length, 1, out);

  /* Write ray tracing metadata to the Track file */
  fwrite(&_num_azim, sizeof(int), 1, out);
  fwrite(&_spacing, sizeof(double), 1, out);
  fwrite(_num_tracks, sizeof(int), _num_azim, out);
  fwrite(_num_x, sizeof(int), _num_azim, out);
  fwrite(_num_y, sizeof(int), _num_azim, out);

  /* Write the azimuthal angle quadrature weights to the Track file */
  double* azim_weights = new double[_num_azim];
  for (int i=0; i < _num_azim; i++)
    azim_weights[i] = _azim_weights[i];
  fwrite(azim_weights, sizeof(double), _num_azim, out);
  free(azim_weights);

  Track* curr_track;
  double x0, y0, x1, y1;
  double phi;
  int azim_angle_index;
  int num_segments;
  std::vector<segment*> _segments;

  segment* curr_segment;
  double length;
  int material_id;
  int region_id;
  int mesh_surface_fwd;
  int mesh_surface_bwd;

  /* Loop over all Tracks */
  for (int i=0; i < _num_azim; i++) {
    for (int j=0; j < _num_tracks[i]; j++) {

      /* Get data for this Track */
      curr_track = &_tracks[i][j];
      x0 = curr_track->getStart()->getX();
      y0 = curr_track->getStart()->getY();
      x1 = curr_track->getEnd()->getX();
      y1 = curr_track->getEnd()->getY();
      phi = curr_track->getPhi();
      azim_angle_index = curr_track->getAzimAngleIndex();
      num_segments = curr_track->getNumSegments();

      /* Write data for this Track to the Track file */
      fwrite(&x0, sizeof(double), 1, out);
      fwrite(&y0, sizeof(double), 1, out);
      fwrite(&x1, sizeof(double), 1, out);
      fwrite(&y1, sizeof(double), 1, out);
      fwrite(&phi, sizeof(double), 1, out);
      fwrite(&azim_angle_index, sizeof(int), 1, out);
      fwrite(&num_segments, sizeof(int), 1, out);

      /* Loop over all segments for this Track */
      for (int s=0; s < num_segments; s++) {

        /* Get data for this segment */
        curr_segment = curr_track->getSegment(s);
        length = curr_segment->_length;
        material_id = curr_segment->_material->getId();
        region_id = curr_segment->_region_id;

        /* Write data for this segment to the Track file */
        fwrite(&length, sizeof(double), 1, out);
        fwrite(&material_id, sizeof(int), 1, out);
        fwrite(&region_id, sizeof(int), 1, out);

        /* Write CMFD-related data for the Track if needed */
        if (_geometry->getMesh()->getCmfdOn()){
          mesh_surface_fwd = curr_segment->_mesh_surface_fwd;
          mesh_surface_bwd = curr_segment->_mesh_surface_bwd;
          fwrite(&mesh_surface_fwd, sizeof(int), 1, out);
          fwrite(&mesh_surface_bwd, sizeof(int), 1, out);
        }
      }
    }
  }

  /* Close the Track file */
  fclose(out);

  /* Inform other the TrackGenerator::generateTracks() method that it may
   * import ray tracing data from this file if it is called and the ray
   * tracing parameters have not changed */
  _use_input_file = true;

  return;
}


/**
 * @brief Reads Tracks in from a "*.tracks" binary file.
 * @details Storing Tracks in a binary file saves time by eliminating ray
 *          tracing for Track segmentation in commonly simulated geometries.
 * @return true if able to read Tracks in from a file; false otherwise
 */
bool TrackGenerator::readTracksFromFile() {

  /* Deletes Tracks arrays if tracks have been generated */
  if (_contains_tracks) {
    delete [] _num_tracks;
    delete [] _num_segments;
    delete [] _num_x;
    delete [] _num_y;
    delete [] _azim_weights;

    for (int i = 0; i < _num_azim; i++)
      delete [] _tracks[i];

    delete [] _tracks;
  }

  int ret;
  FILE* in;
  in = fopen(_tracks_filename.c_str(), "r");

  int string_length;

  /* Import Geometry metadata from the Track file */
  ret = fread(&string_length, sizeof(int), 1, in);
  char* geometry_to_string = new char[string_length];
  ret = fread(geometry_to_string, sizeof(char)*string_length, 1, in);

  /* Check if our Geometry is exactly the same as the Geometry in the
   * Track file for this number of azimuthal angles and track spacing */
  if (_geometry->toString().compare(geometry_to_string) != 0)
    return false;

  delete [] geometry_to_string;

  log_printf(NORMAL_LOG, "Importing ray tracing data from file...");

  /* Import ray tracing metadata from the Track file */
  ret = fread(&_num_azim, sizeof(int), 1, in);
  ret = fread(&_spacing, sizeof(double), 1, in);

  /* Initialize data structures for Tracks */
  _num_tracks = new int[_num_azim];
  _num_x = new int[_num_azim];
  _num_y = new int[_num_azim];
  _azim_weights = new FP_PRECISION[_num_azim];
  double* azim_weights = new double[_num_azim];
  _tracks = new Track*[_num_azim];

  ret = fread(_num_tracks, sizeof(int), _num_azim, in);
  ret = fread(_num_x, sizeof(int), _num_azim, in);
  ret = fread(_num_y, sizeof(int), _num_azim, in);
  ret = fread(azim_weights, sizeof(double), _num_azim, in);

  /* Import azimuthal angle quadrature weights from Track file */
  for (int i=0; i < _num_azim; i++)
    _azim_weights[i] = azim_weights[i];

  free(azim_weights);

  Track* curr_track;
  double x0, y0, x1, y1;
  double phi;
  int azim_angle_index;
  int num_segments;

  double length;
  int material_id;
  int region_id;

  int mesh_surface_fwd;
  int mesh_surface_bwd;

  /* Calculate the total number of Tracks */
  for (int i=0; i < _num_azim; i++)
    _tot_num_tracks += _num_tracks[i];

  /* Allocate memory for the number of segments per Track array */
  _num_segments = new int[_tot_num_tracks];

  int uid = 0;
  _tot_num_segments = 0;

  /* Loop over Tracks */
  for (int i=0; i < _num_azim; i++) {

    _tracks[i] = new Track[_num_tracks[i]];

    for (int j=0; j < _num_tracks[i]; j++) {

      /* Import data for this Track from Track file */
      ret = fread(&x0, sizeof(double), 1, in);
      ret = fread(&y0, sizeof(double), 1, in);
      ret = fread(&x1, sizeof(double), 1, in);
      ret = fread(&y1, sizeof(double), 1, in);
      ret = fread(&phi, sizeof(double), 1, in);
      ret = fread(&azim_angle_index, sizeof(int), 1, in);
      ret = fread(&num_segments, sizeof(int), 1, in);

      _tot_num_segments += num_segments;
      _num_segments[uid] += num_segments;

      /* Initialize a Track with this data */
      curr_track = &_tracks[i][j];
      curr_track->setValues(x0, y0, x1, y1, phi);
      curr_track->setUid(uid);
      curr_track->setAzimAngleIndex(azim_angle_index);

      /* Loop over all segments in this Track */
      for (int s=0; s < num_segments; s++) {

        /* Import data for this segment from Track file */
        ret = fread(&length, sizeof(double), 1, in);
        ret = fread(&material_id, sizeof(int), 1, in);
        ret = fread(&region_id, sizeof(int), 1, in);

        /* Initialize segment with the data */
        segment* curr_segment = new segment;
        curr_segment->_length = length;
        curr_segment->_material = _geometry->getMaterial(material_id);
        curr_segment->_region_id = region_id;

        /* Import CMFD-related data if needed */
        if (_geometry->getMesh()->getCmfdOn()){
          ret = fread(&mesh_surface_fwd, sizeof(int), 1, in);
          ret = fread(&mesh_surface_bwd, sizeof(int), 1, in);
          curr_segment->_mesh_surface_fwd = mesh_surface_fwd;
          curr_segment->_mesh_surface_bwd = mesh_surface_bwd;
        }

        /* Add this segment to the Track */
        curr_track->addSegment(curr_segment);
      }

      uid++;
    }
  }

  /* Inform the rest of the class methods that Tracks have been initialized */
  if (ret)
    _contains_tracks = true;

  /* Close the Track file */
  fclose(in);

  return true;
}
