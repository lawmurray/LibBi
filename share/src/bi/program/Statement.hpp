/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_STATEMENT_HPP
#define BI_PROGRAM_STATEMENT_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Statement.
 *
 * @ingroup program
 */
class Statement : public Expression {
public:
  /**
   * Destructor.
   */
  virtual ~Statement() = 0;
};
}

inline biprog::Statement::~Statement() {
  //
}

#endif

