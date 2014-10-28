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

namespace biprog {
/**
 * Sequence of statements.
 *
 * @ingroup program
 */
class Sequence: public Statement, public boost::enable_shared_from_this<
    Sequence> {
public:
  /**
   * Constructor.
   */
  Sequence(Expression* head, Expression* tail = NULL);

  /**
   * Destructor.
   */
  virtual ~Sequence();

  /**
   * First statement.
   */
  boost::shared_ptr<Expression> head;

  /**
   * Remaining statements.
   */
  boost::shared_ptr<Expression> tail;
};
}

inline biprog::Sequence::Sequence(Expression* head, Expression* tail) :
    head(head), tail(tail) {
  //
}

inline biprog::Sequence::~Sequence() {
  //
}

#endif
