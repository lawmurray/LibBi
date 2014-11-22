/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONAL_HPP
#define BI_PROGRAM_CONDITIONAL_HPP

#include "Typed.hpp"
#include "Conditioned.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public virtual Typed,
    public virtual Conditioned,
    public virtual Braced,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Conditional> {
public:
  /**
   * Constructor.
   */
  Conditional(boost::shared_ptr<Typed> cond, boost::shared_ptr<Typed> braces,
      boost::shared_ptr<Typed> falseBraces, boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Conditional();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

  /**
   * Block if condition is false. May be empty if there is no else clause.
   */
  boost::shared_ptr<Typed> falseBraces;
};
}

inline biprog::Conditional::Conditional(boost::shared_ptr<Typed> cond,
    boost::shared_ptr<Typed> braces, boost::shared_ptr<Typed> falseBraces,
    boost::shared_ptr<Scope> scope) :
    Conditioned(cond), Braced(braces), Scoped(scope), falseBraces(falseBraces) {
  /* pre-condition */
  BI_ASSERT(falseBraces);
}

inline biprog::Conditional::~Conditional() {
  //
}

#endif
