/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EMPTYEXPRESSION_HPP
#define BI_PROGRAM_EMPTYEXPRESSION_HPP

#include "Typed.hpp"

namespace biprog {
/**
 * Empty expression.
 *
 * @ingroup program
 *
 * Used for empty brackets, parentheses or braces.
 */
class EmptyExpression: public virtual Typed,
    public virtual boost::enable_shared_from_this<EmptyExpression> {
public:
  /**
   * Destructor.
   */
  virtual ~EmptyExpression();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual operator bool() const;

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::EmptyExpression::~EmptyExpression() {
  //
}

#endif
