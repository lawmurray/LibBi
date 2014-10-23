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
    public Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param index Statement in square brackets.
   * @param in Statement in parentheses.
   */
  Reference(const char* name, Statement* index = NULL, Statement* in = NULL);

  /**
   * Destructor.
   */
  virtual ~Reference();
};
}

inline biprog::Reference::Reference(const char* name, Statement* index,
    Statement* in) :
    Named(name), Bracketed(index), Parenthesised(in) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif

