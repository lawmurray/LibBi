/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONED_HPP
#define BI_PROGRAM_CONDITIONED_HPP

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Conditioned expression.
 *
 * @ingroup program
 */
class Conditioned {
public:
  /**
   * Constructor.
   *
   * @param cond Conditional expression.
   */
  Conditioned(Expression* cond = NULL);

  /**
   * Destructor.
   */
  virtual ~Conditioned() = 0;

  /**
   * First statement in in brackets.
   */
  boost::scoped_ptr<Expression> cond;
};
}

inline biprog::Conditioned::Conditioned(Expression* cond) :
    cond(cond) {
  //
}

inline biprog::Conditioned::~Conditioned() {
  //
}

#endif

