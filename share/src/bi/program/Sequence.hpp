/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_SEQUENCE_HPP
#define BI_PROGRAM_SEQUENCE_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Sequence of statements.
 *
 * @ingroup program
 */
class Sequence: public virtual Expression,
    public boost::enable_shared_from_this<Sequence> {
public:
  /**
   * Constructor.
   */
  Sequence(boost::shared_ptr<Expression> head,
      boost::shared_ptr<Expression> tail);

  /**
   * Destructor.
   */
  virtual ~Sequence();

  /*
   * Operators.
   */
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

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

inline biprog::Sequence::Sequence(boost::shared_ptr<Expression> head,
    boost::shared_ptr<Expression> tail) :
    head(head), tail(tail) {
  //
}

inline biprog::Sequence::~Sequence() {
  //
}

#endif
