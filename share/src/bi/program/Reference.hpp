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
#include "Bodied.hpp"

#include "boost/shared_ptr.hpp"

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
    public Bodied {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param index Expression in square brackets.
   * @param in Expression in parentheses.
   * @param body Expression in braces.
   */
  Reference(const char* name, Expression* index = NULL, Expression* in = NULL,
      Expression* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Reference();
};
}

inline biprog::Reference::Reference(const char* name, Expression* index,
    Expression* in, Expression* body) :
    Named(name), Bracketed(index), Parenthesised(in), Bodied(body) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif

