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

#ifndef _LOG_H_
#define _LOG_H_     1

//
// 2D MOC neutron transport calculation
// Implement and modification base on the : https://github.com/mit-crpg/OpenMOC
//
/**
 * @file log.h
 * @brief Utility functions for writing log messages to the screen
 * @details Applies level-based logging to print formatted messages
 *          to the screen and to a log file.
 * @author William Boyd (wboyd@mit.edu)
 * @date January 22, 2012
 *
 */

#ifdef __cplusplus
  #include <cstdio>
  #include <cstdarg>
  #include <cstdlib>
  #include <sstream>
  #include <iostream>
  #include <fstream>
  #include <iomanip>
  #include <cstring>
  #include <stdexcept>
  #include <ctime>
  #include <cmath>
  #include <sys/types.h>
  #include <sys/stat.h>
#endif

/**
 * @enum logLevels
 * @brief Logging levels characterize an ordered set of message types
 *        which may be printed to the screen.
 */


/**
 * @var logLevel
 * @brief Logging levels characterize an ordered set of message types
 *        which may be printed to the screen.
 */
typedef enum logLevels {
  /** A debugging message */
  DEBUG_LOG,

  /** An informational but verbose message */
  INFO_LOG,

  /** A brief progress update on run progress */
  NORMAL_LOG,

  /** A message of a single line of characters */
  SEPARATOR_LOG,

  /** A message centered within a line of characters */
  HEADER_LOG,

  /** A message sandwiched between two lines of characters */
  TITLE_LOG,

  /** A message for to warn the user */
  WARNING_LOG,

  /** A message to warn of critical program conditions */
  CRITICAL_LOG,

  /** A message containing program results */
  RESULT_LOG,

  /** A messsage for unit testing */
  UNITTEST_LOG,

  /** A message reporting error conditions */
  ERROR_LOG
} logLevel;


/**
 * @brief A function stub used to convert C++ exceptions into Python exceptions
 *        through SWIG.
 * @details This method is not defined in the C++ source. It is defined in the
 *          SWIG inteface files (i.e., openmoc/openmoc.i)
 * @param msg a character array for the exception message
 */
extern void set_err(const char *msg);

void set_output_directory(char* directory);
const char* get_output_directory();
void set_log_filename(char* filename);
const char* get_log_filename();

void set_separator_character(char c);
char get_separator_character();
void set_header_character(char c);
char get_header_character();
void set_title_character(char c);
char get_title_character();
void set_line_length(int length);
void set_log_level(const char* new_level);
int get_log_level();

void log_printf(logLevel level, const char *format, ...);
std::string create_multiline_msg(std::string level, std::string message);

#endif /* LOG_H_ */
