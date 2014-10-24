/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EXPRESSION_HPP
#define BI_PROGRAM_EXPRESSION_HPP

namespace biprog {
/**
 * Expression.
 *
 * @ingroup program
 */
class Expression {
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

