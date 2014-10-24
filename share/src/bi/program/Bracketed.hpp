/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACKETED_HPP
#define BI_PROGRAM_BRACKETED_HPP

#include "boost/shared_ptr.hpp"

namespace biprog {
/**
 * Bracketed expression.
 *
 * @ingroup program
 */
class Bracketed {
public:
  /**
   * Constructor.
   *
   * @param index Expression in square brackets.
   */
  Bracketed(Expression* index = NULL);

  /**
   * Destructor.
   */
  virtual ~Bracketed() = 0;

  /**
   * First statement in index brackets.
   */
  boost::shared_ptr<Expression> index;
};
}

inline biprog::Bracketed::Bracketed(Expression* index) :
    index(index) {
  //
}

inline biprog::Bracketed::~Bracketed() {
  //
}

#endif

