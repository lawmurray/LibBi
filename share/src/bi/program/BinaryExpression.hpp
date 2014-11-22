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
class BinaryExpression: public virtual Typed,
    public virtual boost::enable_shared_from_this<BinaryExpression> {
public:
  /**
   * Constructor.
   */
  BinaryExpression(boost::shared_ptr<Typed> left, Operator op,
      boost::shared_ptr<Typed> right);

  /**
   * Destructor.
   */
  virtual ~BinaryExpression();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

  /**
   * Left operand.
   */
  boost::shared_ptr<Typed> left;

  /**
   * Operator.
   */
  Operator op;

  /**
   * Right operand.
   */
  boost::shared_ptr<Typed> right;
};
}

inline biprog::BinaryExpression::BinaryExpression(
    boost::shared_ptr<Typed> left, Operator op,
    boost::shared_ptr<Typed> right) :
    left(left), op(op), right(right) {
  /* pre-conditions */
  BI_ASSERT(left);
  BI_ASSERT(right);

  type = left->type;  //@todo Infer type properly
}

inline biprog::BinaryExpression::~BinaryExpression() {
  //
}

#endif
