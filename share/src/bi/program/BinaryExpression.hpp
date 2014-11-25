/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BINARYEXPRESSION_HPP
#define BI_PROGRAM_BINARYEXPRESSION_HPP

#include "Typed.hpp"
#include "Operator.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class BinaryExpression: public virtual Typed {
public:
  /**
   * Constructor.
   */
  BinaryExpression(Typed* left, Operator op, Typed* right);

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Left operand.
   */
  Typed* left;

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

inline biprog::BinaryExpression::BinaryExpression(Typed* left, Operator op,
    Typed* right) :
    Typed(left->type->clone()), left(left), op(op), right(right) {
  /* pre-conditions */
  BI_ASSERT(left);
  BI_ASSERT(right);

  type = left->type;  //@todo Infer type properly
}

inline biprog::BinaryExpression::~BinaryExpression() {
  delete left;
  delete right;
}

#endif
