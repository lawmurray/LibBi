/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_EXCEPTION_UNRESOLVEDREFERENCEEXCEPTION_HPP
#define BI_EXCEPTION_UNRESOLVEDREFERENCEEXCEPTION_HPP

#include "Exception.hpp"
#include "../program/Reference.hpp"

#include "boost/shared_ptr.hpp"

namespace biprog {
/**
 * Unresolved reference in program.
 */
struct UnresolvedReferenceException: public Exception {
  /**
   * Constructor.
   */
  UnresolvedReferenceException(boost::shared_ptr<Reference> ref);

  /**
   * Destructor.
   */
  virtual ~UnresolvedReferenceException();
};
}

inline biprog::UnresolvedReferenceException::~UnresolvedReferenceException() {
  //
}

#endif
