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
class Sequence: public virtual Statement {
public:
  /**
   * Constructor.
   */
  Sequence(Statement* head, Statement* tail);

  /**
   * Destructor.
   */
  virtual ~Sequence();

  virtual Sequence* clone();
  virtual Statement* accept(Visitor& v);

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

  /**
   * Left operand.
   */
  Statement* head;

  /**
   * Right operand.
   */
  Statement* tail;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Sequence::Sequence(Statement* head, Statement* tail) :
    head(head), tail(tail) {
  /* pre-conditions */
  BI_ASSERT(head);
  BI_ASSERT(tail);
}

inline biprog::Sequence::~Sequence() {
  delete head;
  delete tail;
}

#endif
