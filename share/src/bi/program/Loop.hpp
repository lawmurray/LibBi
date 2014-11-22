/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LOOP_HPP
#define BI_PROGRAM_LOOP_HPP

#include "Typed.hpp"
#include "Conditioned.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Loop.
 *
 * @ingroup program
 */
class Loop: public virtual Typed,
    public virtual Conditioned,
    public virtual Braced,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Loop> {
public:
  /**
   * Constructor.
   */
  Loop(boost::shared_ptr<Typed> cond, boost::shared_ptr<Typed> braces,
      boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Loop();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Loop::Loop(boost::shared_ptr<Typed> cond,
    boost::shared_ptr<Typed> braces, boost::shared_ptr<Scope> scope) :
    Conditioned(cond), Braced(braces), Scoped(scope) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

#endif
