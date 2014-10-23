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
#include "Bracketed.hpp"
#include "Expression.hpp"

#include "boost/shared_ptr.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class Reference: public Named, public Bracketed, public Expression {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param in Statement in parentheses.
   * @param index Statement in square brackets.
   * @param body Statement in curly brackets.
   */
  Reference(const char* name, Statement* in = NULL, Statement* index = NULL,
      Statement* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Reference();

  /**
   * First statement in intheses.
   */
  boost::shared_ptr<Statement> in;

  /**
   * First statement in index brackets.
   */
  boost::shared_ptr<Statement> index;

  /**
   * First statement in body brackets.
   */
  boost::shared_ptr<Statement> body;
};
}

inline biprog::Reference::Reference(const char* name, Statement* in,
    Statement* index, Statement* body) :
    Named(name), in(in), index(index), body(body) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif

