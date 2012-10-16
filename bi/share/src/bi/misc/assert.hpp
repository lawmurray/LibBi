/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev $
 * $Date$
 */
#ifndef BI_MISC_ASSERT_HPP
#define BI_MISC_ASSERT_HPP

#include "macro.hpp"

#include <cassert>
#include <iostream>

/**
 * @def BI_ASSERT(cond, msg)
 *
 * Checks condition, terminating if failed. Works in CUDA device code for
 * compute capability 2.0 and later, disabled otherwise. Debugging mode only.
 *
 * @arg @p cond Condition.
 */
#ifndef NDEBUG
#define BI_ASSERT(cond) assert(cond)
#else
#define BI_ASSERT(cond)
#endif

/**
 * @def BI_ASSERT_MSG(cond, msg)
 *
 * Checks condition, terminating and printing error if failed. Debugging
 * mode only.
 *
 * @arg @p cond Condition.
 * @arg @p msg Message to print if condition failed.
 */
#ifndef NDEBUG
#define BI_ASSERT_MSG(cond, msg) \
  if (!(cond)) { \
    std::cerr << "Error: " << msg << std::endl; \
  } \
  assert(cond)
#else
#define BI_ASSERT_MSG(cond, msg)
#endif

/**
 * @def BI_ERROR_MSG(cond, msg)
 *
 * Checks condition, terminating and printing error if failed.
 *
 * @arg @p cond Condition.
 * @arg @p msg Message to print if condition failed.
 */
#ifdef NDEBUG
#define BI_ERROR_MSG(cond, msg) \
  if (!(cond)) { \
    std::cerr << "Error: " << msg << std::endl; \
    exit(1); \
  }
#else
#define BI_ERROR_MSG(cond, msg) BI_ASSERT_MSG(cond, msg)
#endif

/**
 * @def BI_WARN_MSG
 *
 * Checks condition, printing warning if failed.
 *
 * @arg @p cond Condition.
 * @arg @p msg Message to print.
 */
#define BI_WARN_MSG(cond, msg) \
  if (!(cond)) { \
    std::cerr << "Warning: " << msg << std::endl; \
  }

#endif
