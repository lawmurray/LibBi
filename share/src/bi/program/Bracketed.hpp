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
   * @param index Statement in square brackets.
   */
  Bracketed(Statement* index = NULL);

  /**
   * Destructor.
   */
  virtual ~Bracketed() = 0;

  /**
   * First statement in index brackets.
   */
  boost::shared_ptr<Statement> index;
};
}

inline biprog::Bracketed::Bracketed(Statement* index) :
    index(index) {
  //
}

inline biprog::Bracketed::~Bracketed() {
  //
}

#endif

