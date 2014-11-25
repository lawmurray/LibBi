/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SEQUENCE_HPP
#define BI_PROGRAM_SEQUENCE_HPP

#include "Typed.hpp"

namespace biprog {
/**
 * Sequence of statements.
 *
 * @ingroup program
 */
class Sequence: public virtual Typed {
public:
  /**
   * Constructor.
   */
  Sequence(Typed* head, Typed* tail);

  /**
   * Destructor.
   */
  virtual ~Sequence();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Left operand.
   */
  Typed* head;

  /**
   * Right operand.
   */
  Typed* tail;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Sequence::Sequence(Typed* head, Typed* tail) :
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
