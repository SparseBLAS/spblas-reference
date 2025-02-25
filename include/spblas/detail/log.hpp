#pragma once

//
// This defines a logging system that can be turned on and off by command line
//
//  -DLOG_LEVEL=SPBLAS_TRACE   (or any other level or undefined LOG_LEVEL to
//  turn it off)
//
// SPBLAS_DEBUG   (level 0) not meant to be kept in code, but good for debugging
// SPBLAS_WARNING (level 1) for giving developers a more complete warning
//                          message before throwing error or exiting
// SPBLAS_TRACE   (level 2) provides trace comments of files and line numbers
// SPBLAS_INFO    (level 3) for providing deep information about algorithm or
//                          details
//
//
//  Add to code any of the following
//
//  log_debug("formatted message"); // behaves like printf so can add formatting
//  log_warning("formated message"); // warnings or early exit extra detail
//  log_trace(""); // can also add formatted message, but often file/line is
//                 // sufficient
//  log_info("formatted message") // any extra info or data we find useful for
//                                // analyzing algorithm or debugging
//
//  when LOG_LEVEL is not defined, it does nothing, but when defined, it prints
//  as desired for all levels up to specified one
//

enum { SPBLAS_DEBUG, SPBLAS_WARNING, SPBLAS_TRACE, SPBLAS_INFO };

#define log_debug(fmt, ...)                                                    \
  _log_(SPBLAS_DEBUG, __FILE__, __LINE__, __PRETTY_FUNCTION__, fmt,            \
        ##__VA_ARGS__)
#define log_warning(fmt, ...)                                                  \
  _log_(SPBLAS_WARNING, __FILE__, __LINE__, __PRETTY_FUNCTION__, fmt,          \
        ##__VA_ARGS__)
#define log_trace(fmt, ...)                                                    \
  _log_(SPBLAS_TRACE, __FILE__, __LINE__, __PRETTY_FUNCTION__, fmt,            \
        ##__VA_ARGS__)
#define log_info(fmt, ...)                                                     \
  _log_(SPBLAS_INFO, __FILE__, __LINE__, __PRETTY_FUNCTION__, fmt,             \
        ##__VA_ARGS__)

#if defined(LOG_LEVEL)
#define _log_(l, file, line, func, fmt, ...)                                   \
  spblas_log_(l, #l, file, line, func, fmt, ##__VA_ARGS__)
#else
#define _log_(l, file, line, func, fmt, ...)                                   \
  do {                                                                         \
  } while (0)
#endif

#ifdef LOG_LEVEL

#include <stdarg.h> // va_start, va_list, va_end
#include <stdio.h>  // printf, vprintf

static void spblas_log_(int level, const char* pref, const char* file,
                        const int line, const char* func, const char* fmt,
                        ...) {
  va_list args;
  va_start(args, fmt);

  if (level <= LOG_LEVEL) { // add all smaller logtype enums

    if (isatty(1)) {
      // color log if isatty(1) ie file descriptor is from terminal
      printf("\x1b[48;5;14m"); // background is high intensity light blue
      printf("\x1b[38;5;0m");  // foreground is black
    }

    // print out preamble:  [logtype] file:<line#>: functionname() message
    // printf("[%s] %s:%d: %s()", pref, file, line, func);

    // print out preamble:  [logtype] file:<line#>: message
    printf("[%s] %s:%d: ", pref, file, line);

    vprintf(fmt, args); // print out message

    // end color for log printing
    if (isatty(1)) {
      printf("\x1b[0m");
    }
    printf("\n");
  } // if level <= LOG_LEVEL
  fflush(0);
  va_end(args);
}
#endif // LOG_LEVEL
