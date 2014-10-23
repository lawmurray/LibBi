/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SEQUENCE_HPP
#define BI_PROGRAM_SEQUENCE_HPP

#include "Statement.hpp"

#include "boost/shared_ptr.hpp"

namespace biprog {
/**
 * Sequence of statements.
 *
 * @ingroup program
 */
class Sequence: public Statement {
public:
  /**
   * Constructor.
   */
  Sequence(Statement* head, Statement* tail = NULL);

  /**
   * Destructor.
   */
  virtual ~Sequence();

  /**
   * First statement.
   */
  boost::shared_ptr<Statement> head;

  /**
   * Remaining statements.
   */
  boost::shared_ptr<Statement> tail;
};
}

inline biprog::Sequence::Sequence(Statement* head, Statement* tail) :
    head(head), tail(tail) {
  //
}

inline biprog::Sequence::~Sequence() {
  //
}

#endif
