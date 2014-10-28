/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_REFERENCE_HPP
#define BI_PROGRAM_REFERENCE_HPP

#include "Expression.hpp"
#include "Named.hpp"
#include "Bracketed.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class Reference: public Expression,
    public Named,
    public Bracketed,
    public Parenthesised,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param brackets Expression in square brackets.
   * @param parens Expression in parentheses.
   * @param braces Expression in braces.
   */
  Reference(const char* name, Expression* brackets = NULL,
      Expression* parens = NULL, Expression* braces = NULL);

  /**
   * Destructor.
   */
  virtual ~Reference();
};
}

inline biprog::Reference::Reference(const char* name, Expression* brackets,
    Expression* parens, Expression* braces) :
    Named(name), Bracketed(brackets), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif

