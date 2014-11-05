/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LOOP_HPP
#define BI_PROGRAM_LOOP_HPP

#include "Conditioned.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Loop.
 *
 * @ingroup program
 */
class Loop: public virtual Conditioned,
    public virtual Braced,
    public virtual Scoped,
    public boost::enable_shared_from_this<Loop> {
public:
  /**
   * Constructor.
   */
  Loop(boost::shared_ptr<Expression> cond,
      boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Loop();

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
};
}

inline biprog::Loop::Loop(boost::shared_ptr<Expression> cond,
    boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope) :
    Conditioned(cond), Braced(braces), Scoped(scope) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

#endif
