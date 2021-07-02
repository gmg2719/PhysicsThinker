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

#ifndef _MY_TIMER_H_
#define _MY_TIMER_H_      1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//

/**
 * @file Timer.h
 * @brief The Timer class.
 * @date January 2, 2012
 * @author  William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifdef __cplusplus
  #include <ctime>
  #include <iostream>
  #include <sstream>
  #include <iomanip>
  #include <utility>
  #include <map>
  #include <vector>
  #include <string>
  #include <sys/time.h>
#ifdef _OPENMP
  #include <omp.h>
#endif
  #include "common/log.h"
#endif


/**
 * @class Timer Timer.h "src/Timer.cpp"
 * @brief The Timer class is for timing and profiling regions of code.
 */
class Timer {

private:

  /** A vector of floating point start times at each inclusive level
   *  at which we are timing */
  static std::vector<double> _start_times;

  /** The time elapsed (seconds) for the current split */
  float _elapsed_time;

  /** Whether or not the Timer is running for the current split */
  bool _running;

  /** A vector of the times and messages for each split */
  static std::map<std::string, double> _timer_splits;

  /**
   * @brief Assignment operator for static referencing of the Timer.
   * @param & the Timer static class object
   * @return a pointer to the Timer static class object
   */
  Timer &operator=(const Timer &) { return *this; }

  /**
   * @brief Timer constructor.
   * @param & The Timer static reference pointer.
   */
  Timer(const Timer &) { }

public:
  /**
   * @brief Constructor sets the current split elapsed time to zero.
   */
  Timer() {
    _running = false;
    _elapsed_time = 0;
  }

  /**
   * @brief Destructor
   */
  virtual ~Timer() { }

  /**
   * @brief Returns a static instance of the Timer class.
   * @return a pointer to the static Timer class
   */
  static Timer *Get() {
    static Timer instance;
    return &instance;
  }

  void startTimer();
  void stopTimer();
  void recordSplit(const char* msg);
  double getTime();
  double getSplit(const char* msg);
  void printSplits();
  void clearSplit(const char* msg);
  void clearSplits();
};

#endif /* TIMER_H_ */
