/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

#include "Statement.hpp"

namespace biprog {
/**
 * Expression.
 *
 * @ingroup program
 */
class Expression: public Statement {
public:
  /**
   * Constructor.
   */
  Expression();

  /**
   * Destructor.
   */
  virtual ~Expression();
};
}

#endif

