/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_RETURNEXPRESSION_HPP
#define BI_PROGRAM_RETURNEXPRESSION_HPP

#include "Typed.hpp"

namespace biprog {
/**
 * Unary expression.
 *
 * @ingroup program
 */
class ReturnExpression: public virtual Typed,
    public virtual boost::enable_shared_from_this<ReturnExpression> {
public:
  /**
   * Constructor.
   *
   * @param expr Right operand.
   */
  ReturnExpression(boost::shared_ptr<Typed> expr);

  /**
   * Destructor.
   */
  virtual ~ReturnExpression();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Right operand.
   */
  boost::shared_ptr<Typed> expr;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::ReturnExpression::ReturnExpression(
    boost::shared_ptr<Typed> expr) :
    Typed(expr->type), expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::ReturnExpression::~ReturnExpression() {
  //
}

#endif
