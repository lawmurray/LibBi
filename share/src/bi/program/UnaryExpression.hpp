/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_UNARYEXPRESSION_HPP
#define BI_PROGRAM_UNARYEXPRESSION_HPP

#include "Typed.hpp"
#include "Operator.hpp"

namespace biprog {
/**
 * Unary expression.
 *
 * @ingroup program
 */
class UnaryExpression: public virtual Typed {
public:
  /**
   * Constructor.
   *
   * @param op Operator.
   * @param right Right operand.
   */
  UnaryExpression(Operator op, Typed* right);

  /**
   * Destructor.
   */
  virtual ~UnaryExpression();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Operator.
   */
  Operator op;

  /**
   * Right operand.
   */
  Typed* right;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::UnaryExpression::UnaryExpression(Operator op, Typed* right) :
    Typed(right->type->clone()), op(op), right(right) {
  /* pre-condition */
  BI_ASSERT(right);

  type = right->type;  //@todo Infer type properly
}

inline biprog::UnaryExpression::~UnaryExpression() {
  delete right;
}

#endif
