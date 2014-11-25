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
class EmptyExpression: public virtual Typed {
public:
  /**
   * Destructor.
   */
  virtual ~EmptyExpression();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

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
