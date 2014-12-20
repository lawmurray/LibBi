/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_EXCEPTION_EXCEPTION_HPP
#define BI_EXCEPTION_EXCEPTION_HPP

#include <string>

namespace biprog {
/**
 * Exception.
 */
struct Exception {
  /**
   * Destructor.
   */
  virtual ~Exception() = 0;

  /**
   * Message.
   */
  std::string msg;
};
}

inline biprog::Exception::~Exception() {
  //
}

#endif
