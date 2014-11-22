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
class Sequence: public virtual Typed,
    public virtual boost::enable_shared_from_this<Sequence> {
public:
  /**
   * Constructor.
   */
  Sequence(boost::shared_ptr<Typed> head, boost::shared_ptr<Typed> tail);

  /**
   * Destructor.
   */
  virtual ~Sequence();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

  /**
   * Left operand.
   */
  boost::shared_ptr<Typed> head;

  /**
   * Right operand.
   */
  boost::shared_ptr<Typed> tail;
};
}

inline biprog::Sequence::Sequence(boost::shared_ptr<Typed> head,
    boost::shared_ptr<Typed> tail) :
    head(head), tail(tail) {
  /* pre-conditions */
  BI_ASSERT(head);
  BI_ASSERT(tail);
}

inline biprog::Sequence::~Sequence() {
  //
}

#endif
