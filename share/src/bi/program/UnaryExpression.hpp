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
class UnaryExpression: public virtual Typed,
    public virtual boost::enable_shared_from_this<UnaryExpression> {
public:
  /**
   * Constructor.
   *
   * @param op Operator.
   * @param right Right operand.
   */
  UnaryExpression(Operator op, boost::shared_ptr<Typed> right);

  /**
   * Destructor.
   */
  virtual ~UnaryExpression();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

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

inline biprog::UnaryExpression::UnaryExpression(Operator op,
    boost::shared_ptr<Typed> right) :
    op(op), right(right) {
  /* pre-condition */
  BI_ASSERT(right);

  setType(right->type);  //@todo Infer type properly
}

inline biprog::UnaryExpression::~UnaryExpression() {
  //
}

#endif
