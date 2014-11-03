/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "boost/shared_ptr.hpp"
#include "boost/enable_shared_from_this.hpp"

namespace biprog {
/**
 * Expression.
 *
 * @ingroup program
 */
class Expression: public boost::enable_shared_from_this<Expression> {
public:

  /**
   * Destructor.
   */
  virtual ~Expression() = 0;
};
}

inline biprog::Expression::~Expression() {
  //
}

#endif

