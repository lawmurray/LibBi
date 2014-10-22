/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_REFERENCE_HPP
#define BI_PROGRAM_REFERENCE_HPP

#include "Named.hpp"
#include "Expression.hpp"

#include "boost/shared_ptr.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class Reference: public Named, public Expression {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param paren First statement in parentheses.
   * @param square First statement in square brackets.
   * @param curly First statement in curly brackets.
   */
  Reference(const char* name, const Statement* paren = NULL,
      const Statement* square = NULL, const Statement* curly = NULL);

  /**
   * Destructor.
   */
  virtual ~Reference();

  /**
   * First statement in parentheses.
   */
  boost::shared_ptr<Statement> paren;

  /**
   * First statement in square brackets.
   */
  boost::shared_ptr<Statement> square;

  /**
   * First statement in curly brackets.
   */
  boost::shared_ptr<Statement> curly;
};
}

inline biprog::Reference::Reference(const char* name, const Statement* paren,
    const Statement* square, const Statement* curly) :
    Named(name), paren(paren), square(square), curly(curly) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif

