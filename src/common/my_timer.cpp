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

#include "common/my_timer.h"


std::map<std::string, double> Timer::_timer_splits;
std::vector<double> Timer::_start_times;

/**
 * @brief Starts the Timer.
 * @details This method is similar to starting a stopwatch.
 */
void Timer::startTimer() {
#ifdef _OPENMP
  double start_time = omp_get_wtime();
#else
  struct timeval t;
  gettimeofday(&t, NULL);

  double start_time = t.tv_sec + (double)t.tv_usec / 1000000ULL;
#endif
  _start_times.push_back(start_time);
  _running = true;

  return;
}


/**
 * @brief Stops the Timer.
 * @details This method is similar to stopping a stopwatch.
 */
void Timer::stopTimer() {

  if (_running) {
#ifdef _OPENMP
    double end_time = omp_get_wtime();
#else
  struct timeval t;
  gettimeofday(&t, NULL);

  double end_time = t.tv_sec + (double)t.tv_usec / 1000000ULL;
#endif
    double start_time = _start_times.back();

    _elapsed_time = end_time - start_time;

    if (_start_times.empty())
      _running = false;

    _start_times.pop_back();
  }

  return;
}


/**
 * @brief Records a message corresponding to a time for the current split.
 * @details When this method is called it assumes that the Timer has been
 *          stopped and has the current time for the process corresponding
 *          to the message.
 * @param msg a msg corresponding to this time split
 */
void Timer::recordSplit(const char* msg) {

  double time = getTime();
  std::string msg_string = std::string(msg);

  if (_timer_splits.find(msg_string) != _timer_splits.end())
    _timer_splits[msg_string] += time;
  else
    _timer_splits.insert(std::pair<std::string, double>(msg_string, time));
}


/**
 * @brief Returns the time elapsed from startTimer() to stopTimer().
 * @return the elapsed time in seconds
 */
double Timer::getTime() {
  return _elapsed_time;
}


/**
 * @brief Returns the time associated with a particular split.
 * @details If the split does not exist, returns 0.
 * @param msg the message tag for the split
 * @return the time recorded for the split (seconds)
 */
double Timer::getSplit(const char* msg) {

  std::string msg_string = std::string(msg);

  if (_timer_splits.find(msg_string) == _timer_splits.end())
    return 0.0;
  else
    return _timer_splits[msg_string];
}


/**
 * @brief Prints the times and messages for each split to the console.
 * @details This method will loop through all of the Timer's splits and print a
 *          formatted message string (80 characters in length) to the console
 *          with the message and the time corresponding to that message.
 */
void Timer::printSplits() {

  std::string curr_msg;
  double curr_split;
  std::map<std::string, double>::iterator iter;

  for (iter = _timer_splits.begin(); iter != _timer_splits.end(); ++iter) {

    std::stringstream formatted_msg;

    curr_msg = (*iter).first;
    curr_split = (*iter).second;

    curr_msg.resize(53, '.');
    formatted_msg << curr_msg;

    log_printf(RESULT, "%s%1.4E sec", formatted_msg.str().c_str(), curr_split);
  }
}


/**
 * @brief Clears the time split for this message and deletes the message's
 *        entry in the Timer's splits log.
 * @param msg the message tag for the split
 */
void Timer::clearSplit(const char* msg) {

  std::string msg_string = std::string(msg);

  if (_timer_splits.find(msg_string) == _timer_splits.end())
    return;
  else
    _timer_splits.erase(msg_string);
}


/**
 * @brief Clears all times split messages from the Timer.
 */
void Timer::clearSplits() {
  _timer_splits.clear();
}
