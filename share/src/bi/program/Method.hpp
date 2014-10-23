/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_METHOD_HPP
#define BI_PROGRAM_METHOD_HPP

#include "Declaration.hpp"

namespace biprog {
/**
 * Method.
 *
 * @ingroup program
 */
class Method: public Declaration {
public:
  /**
   * Constructor.
   */
  Method(Reference* ref);

  /**
   * Destructor.
   */
  virtual ~Method();
};
}

inline biprog::Method::Method(Reference* ref) : Declaration(ref) {
  //
}

inline biprog::Method::~Method() {
  //
}

#endif

