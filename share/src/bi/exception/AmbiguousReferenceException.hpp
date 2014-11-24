/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_EXCEPTION_AMBIGUOUSREFERENCEEXCEPTION_HPP
#define BI_EXCEPTION_AMBIGUOUSREFERENCEEXCEPTION_HPP

#include "Exception.hpp"
#include "../program/Reference.hpp"

#include "boost/shared_ptr.hpp"

namespace biprog {
/**
 * Ambiguous reference in program.
 */
struct AmbiguousReferenceException : public Exception {
  /**
   * Constructor.
   */
  template<class Container>
  AmbiguousReferenceException(boost::shared_ptr<Reference> ref,
      Container matches);

  /**
   * Destructor.
   */
  virtual ~AmbiguousReferenceException();
};
}

#include "boost/typeof/typeof.hpp"

#include <sstream>

template<class Container>
biprog::AmbiguousReferenceException::AmbiguousReferenceException(
    boost::shared_ptr<Reference> ref, Container matches) {
  std::stringstream buf;
  buf << "ambiguous reference " << ref << "; candidates are:" << std::endl;
  BOOST_AUTO(iter, matches.begin());
  for (; iter != matches.end(); ++iter) {
    buf << "  " << **iter << std::endl;
  }
  msg = buf.str();
}

inline biprog::AmbiguousReferenceException::~AmbiguousReferenceException() {
  //
}

#endif
