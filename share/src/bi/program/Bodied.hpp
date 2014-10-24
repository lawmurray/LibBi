/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BODIED_HPP
#define BI_PROGRAM_BODIED_HPP

#include "boost/scoped_ptr.hpp"

namespace biprog {
/**
 * Bodied expression.
 *
 * @ingroup program
 */
class Bodied {
public:
  /**
   * Constructor.
   *
   * @param in Statement in curly brackets.
   */
  Bodied(Expression* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Bodied() = 0;

  /**
   * First statement in in brackets.
   */
  boost::scoped_ptr<Expression> body;
};
}

inline biprog::Bodied::Bodied(Expression* body) :
    body(body) {
  //
}

inline biprog::Bodied::~Bodied() {
  //
}

#endif

