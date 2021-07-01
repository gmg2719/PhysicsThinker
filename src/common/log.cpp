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

#ifdef __cplusplus
  #include "common/log.h"
#endif

/**
 * @var log_level
 * @brief Minimum level of logging messages printed to the screen and log file.
 * @details The default logging level is NORMAL.
 */
static logLevel log_level = NORMAL_LOG;


/**
 * @var log_filename
 * @brief The name of the output log file. By default this is an empty character
 *        array and must be set for messages to be redirected to a log file.
 */
static std::string log_filename = "";


/**
 * @var output_directory
 * @brief The directory in which a "log" folder will be created for logfiles.
 * @details By default this is the same directory as that where the logger
 *          is invoked by an input file.
 */
static std::string output_directory = ".";


/**
 * @var logging
 * @brief A switch which is set to true once the first message is logged.
 * @details The logging switch is needed to indicate whether the output
 *          log file has been created or not.
 */
static bool logging = false;


/**
 * @var separator_char
 * @brief The character to use for SEPARATOR log messages. The default is "-".
 */
static char separator_char = '*';


/**
 * @var header_char
 * @brief The character to use for HEADER log messages. The default is "*".
 */
static char header_char = '*';


/**
 * @var title_char
 * @brief The character to use for TITLE log messages. The default is "*".
 */
static char title_char = '*';


/**
 * @var line_length
 * @brief The maximum line length for a log message.
 * @details The default is 67 which contributes to the standard 80 characters
 *          when accounting for the log level prepended to each log message.
 */
static int line_length = 67;


/**
 * @brief Sets the output directory for log files.
 * @details If the directory does not exist, it creates it for the user.
 * @param directory a character array for the log file directory
 */
void set_output_directory(char* directory) {

  output_directory = std::string(directory);

  /* Check to see if directory exists - if not, create it */
  struct stat st;
  if (!stat(directory, &st) == 0) {
    mkdir(directory, S_IRWXU);
    mkdir((output_directory+"/log").c_str(), S_IRWXU);
  }

  return;
}


/**
 * @brief Returns the output directory for log files.
 * @return a character array for the log file directory
 */
const char* get_output_directory() {
  return output_directory.c_str();
}


/**
 * @brief Sets the name for the log file.
 * @param filename a character array for log filename
 */
void set_log_filename(char* filename) {
  log_filename = std::string(filename);
}


/**
 * @brief Returns the log filename.
 * @return a character array for the log filename
 */
const char* get_log_filename() {
  return log_filename.c_str();
}


/**
 * @brief Sets the character to be used when printing SEPARATOR log messages.
 * @param c the character for SEPARATOR log messages
 */
void set_separator_character(char c) {
  separator_char = c;
}


/**
 * @brief Returns the character used to format SEPARATOR log messages.
 * @return the character used for SEPARATOR log messages
 */
char get_separator_character() {
  return separator_char;
}

/**
 * @brief Sets the character to be used when printing HEADER log messages.
 * @param c the character for HEADER log messages
 */
void set_header_character(char c) {
  header_char = c;
}


/**
 * @brief Returns the character used to format HEADER type log messages.
 * @return the character used for HEADER type log messages
 */
char get_header_character() {
  return header_char;
}


/**
 * @brief Sets the character to be used when printing TITLE log messages.
 * @param c the character for TITLE log messages
 */
void set_title_character(char c) {
  title_char = c;
}


/**
 * @brief Returns the character used to format TITLE log messages.
 * @return the character used for TITLE log messages
 */
char get_title_character() {
  return title_char;
}


/**
 * @brief Sets the maximum line length for log messages.
 * @details Messages longer than this amount will be broken up into
            multiline messages.
 * @param length the maximum log message line length in characters
 */
void set_line_length(int length) {
  line_length = length;
}


/**
 * @brief Sets the minimum log message level which will be printed to the
 *        console and to the log file.
 * @param new_level the minimum logging level as a character array
 */
void set_log_level(const char* new_level) {

  if (strcmp("DEBUG_LOG", new_level) == 0) {
    log_level = DEBUG_LOG;
    log_printf(INFO_LOG, "Logging level set to DEBUG");
  }
  else if (strcmp("INFO_LOG", new_level) == 0) {
    log_level = INFO_LOG;
    log_printf(INFO_LOG, "Logging level set to INFO");
  }
  else if (strcmp("NORMAL_LOG", new_level) == 0) {
    log_level = NORMAL_LOG;
    log_printf(INFO_LOG, "Logging level set to NORMAL");
  }
  else if (strcmp("SEPARATOR_LOG", new_level) == 0) {
    log_level = HEADER_LOG;
    log_printf(INFO_LOG, "Logging level set to SEPARATOR");
  }
  else if (strcmp("HEADER_LOG", new_level) == 0) {
    log_level = HEADER_LOG;
    log_printf(INFO_LOG, "Logging level set to HEADER");
  }
  else if (strcmp("TITLE_LOG", new_level) == 0) {
    log_level = TITLE_LOG;
    log_printf(INFO_LOG, "Logging level set to TITLE");
  }
  else if (strcmp("WARNING_LOG", new_level) == 0) {
    log_level = WARNING_LOG;
    log_printf(INFO_LOG, "Logging level set to WARNING");
  }
  else if (strcmp("CRITICAL_LOG", new_level) == 0) {
    log_level = CRITICAL_LOG;
    log_printf(INFO_LOG, "Logging level set to CRITICAL");
  }
  else if (strcmp("RESULT_LOG", new_level) == 0) {
    log_level = RESULT_LOG;
    log_printf(INFO_LOG, "Logging level set to RESULT");
  }
  else if (strcmp("UNITTEST_LOG", new_level) == 0) {
    log_level = UNITTEST_LOG;
    log_printf(INFO_LOG, "Logging level set to UNITTEST");
  }
  else if (strcmp("ERROR_LOG", new_level) == 0) {
      log_level = ERROR_LOG;
      log_printf(INFO_LOG, "Logging level set to ERROR");
  }

  return;
}


/**
 * @brief Return the minimum level for log messages printed to the screen.
 * @return the minimum level for log messages
 */
int get_log_level(){
  return log_level;
}


/**
 * @brief Print a formatted message to the console.
 * @details If the logging level is ERROR, this function will throw a
 *          runtime exception
 * @param level the logging level for this message
 * @param format variable list of C++ formatted arguments
 */
void log_printf(logLevel level, const char* format, ...) {

  char message[1024];
  std::string msg_string;
  if (level >= log_level) {
    va_list args;

    va_start(args, format);
    vsprintf(message, format, args);
    va_end(args);

    /* Append the log level to the message */
    switch (level) {
    case (DEBUG_LOG):
      {
        std::string msg = std::string(message);
        std::string level_prefix = "[  DEBUG  ]  ";

        /* If message is too long for a line, split into many lines */
        if (int(msg.length()) > line_length)
          msg_string = create_multiline_msg(level_prefix, msg);

        /* Puts message on single line */
        else
          msg_string = level_prefix + msg + "\n";

        break;
      }
    case (INFO_LOG):
      {
        std::string msg = std::string(message);
        std::string level_prefix = "[  INFO   ]  ";

        /* If message is too long for a line, split into many lines */
        if (int(msg.length()) > line_length)
          msg_string = create_multiline_msg(level_prefix, msg);

        /* Puts message on single line */
        else
          msg_string = level_prefix + msg + "\n";

        break;
      }
    case (NORMAL_LOG):
      {
        std::string msg = std::string(message);
        std::string level_prefix = "[  NORMAL ]  ";

        /* If message is too long for a line, split into many lines */
        if (int(msg.length()) > line_length)
          msg_string = create_multiline_msg(level_prefix, msg);

        /* Puts message on single line */
        else
          msg_string = level_prefix + msg + "\n";

        break;
      }
    case (SEPARATOR_LOG):
      {
        std::string pad = std::string(line_length, separator_char);
        std::string prefix = std::string("[SEPARATOR]  ");
        std::stringstream ss;
        ss << prefix << pad << "\n";
        msg_string = ss.str();
        break;
      }
    case (HEADER_LOG):
      {
        int size = strlen(message);
        int halfpad = (line_length - 4 - size) / 2;
        std::string pad1 = std::string(halfpad, header_char);
        std::string pad2 = std::string(halfpad +
                           (line_length - 4 - size) % 2, header_char);
        std::string prefix = std::string("[  HEADER ]  ");
        std::stringstream ss;
        ss << prefix << pad1 << "  " << message << "  " << pad2 << "\n";
        msg_string = ss.str();
        break;
      }
    case (TITLE_LOG):
      {
        int size = strlen(message);
        int halfpad = (line_length - size) / 2;
        std::string pad = std::string(halfpad, ' ');
        std::string prefix = std::string("[  TITLE  ]  ");
        std::stringstream ss;
        ss << prefix << std::string(line_length, title_char) << "\n";
        ss << prefix << pad << message << pad << "\n";
        ss << prefix << std::string(line_length, title_char) << "\n";
        msg_string = ss.str();
        break;
      }
    case (WARNING_LOG):
      {
        std::string msg = std::string(message);
        std::string level_prefix = "[ WARNING ]  ";

        /* If message is too long for a line, split into many lines */
        if (int(msg.length()) > line_length)
          msg_string = create_multiline_msg(level_prefix, msg);

        /* Puts message on single line */
        else
          msg_string = level_prefix + msg + "\n";

        break;
      }
    case (CRITICAL_LOG):
      {
        std::string msg = std::string(message);
        std::string level_prefix = "[ CRITICAL]  ";

        /* If message is too long for a line, split into many lines */
        if (int(msg.length()) > line_length)
          msg_string = create_multiline_msg(level_prefix, msg);

        /* Puts message on single line */
        else
          msg_string = level_prefix + msg + "\n";

        break;
      }
    case (RESULT_LOG):
      msg_string = std::string("[  RESULT ]  ") + message + "\n";
      break;
    case (UNITTEST_LOG):
      {
        std::string msg = std::string(message);
        std::string level_prefix = "[   TEST  ]  ";

        /* If message is too long for a line, split into many lines */
        if (int(msg.length()) > line_length)
          msg_string = create_multiline_msg(level_prefix, msg);

        /* Puts message on single line */
        else
          msg_string = level_prefix + msg + "\n";

        break;
      }
    case (ERROR_LOG):
      {
        /* Create message based on runtime error stack */
        va_start(args, format);
        vsprintf(message, format, args);
        va_end(args);
        std::string msg = std::string(message);
        std::string level_prefix = "[  ERROR  ]  ";

        /* If message is too long for a line, split into many lines */
        if (int(msg.length()) > line_length)
          msg_string = create_multiline_msg(level_prefix, msg);

        /* Puts message on single line */
        else
          msg_string = level_prefix + msg + "\n";
      }
    }

    /* If this is our first time logging, add a header with date, time */
    if (!logging) {

      /* If output directory was not defined by user, then log file is
       * written to a "log" subdirectory. Create it if it doesn't exist */
      if (output_directory.compare(".") == 0) {
        struct stat st;
        if (!stat("log", &st) == 0)
          mkdir("log", S_IRWXU);
      }

      /* Write the message to the output file */
      std::ofstream log_file;
      log_file.open((output_directory + "/" + log_filename).c_str(),
                   std::ios::app);

      /* Append date, time to the top of log output file */
      time_t rawtime;
      struct tm * timeinfo;
      time (&rawtime);
      timeinfo = localtime (&rawtime);
      log_file << "Current local time and date: " << asctime(timeinfo);
      logging = true;

      log_file.close();
    }

    /* Write the log message to the log_file */
    std::ofstream log_file;
    log_file.open((output_directory + "/" + log_filename).c_str(),
                  std::ios::app);
    log_file << msg_string;
    log_file.close();

    /* Write the log message to the shell */
    std::cout << msg_string;
  }


  /* If the message was a runtime error, exit the program */
  if (level == ERROR_LOG)
    exit(1);
}


/**
 * @brief Breaks up a message which is too long for a single line into a
 *        multiline message.
 * @details This is an internal function which is called by log_printf and
 *          should not be called directly by the user.
 * @param level a string containing log level prefix
 * @param message a string containing the log message
 * @return a string with a formatted multiline message
 */
std::string create_multiline_msg(std::string level, std::string message) {

  int size = message.length();

  std::string substring;
  int start = 0;
  int end = line_length;

  std::string msg_string;

  /* Loop over msg creating substrings for each line */
  while (end < size + line_length) {

    /* Append log level to the beginning of each line */
    msg_string += level;

    /* Begin multiline messages with ellipsis */
    if (start != 0)
      msg_string += "... ";

    /* Find the current full length substring for line*/
    substring = message.substr(start, line_length);

    /* Truncate substring to last complete word */
    if (end < size-1) {
      int endspace = substring.find_last_of(" ");
      if (message.at(endspace+1) != ' ' &&
          endspace != int(std::string::npos)) {
        end -= line_length - endspace;
        substring = message.substr(start, end-start);
      }
    }

    /* concatenate substring to output message */
    msg_string += substring + "\n";

    /* Reduce line length to account for ellipsis prefix */
    if (start == 0)
      line_length -= 4;

    /* Update substring indices */
    start = end + 1;
    end += line_length + 1;
  }

  /* Reset line length */
  line_length += 4;

  return msg_string;
}
