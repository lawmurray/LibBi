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

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public virtual Typed,
    public virtual Conditioned,
    public virtual Braced,
    public virtual boost::enable_shared_from_this<Conditional> {
public:
  /**
   * Constructor.
   */
  Conditional(boost::shared_ptr<Typed> cond, boost::shared_ptr<Typed> braces,
      boost::shared_ptr<Typed> falseBraces);

  /**
   * Destructor.
   */
  virtual ~Conditional();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

  /**
   * Block if condition is false. May be empty if there is no else clause.
   */
  boost::shared_ptr<Typed> falseBraces;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Conditional::Conditional(boost::shared_ptr<Typed> cond,
    boost::shared_ptr<Typed> braces, boost::shared_ptr<Typed> falseBraces) :
    Conditioned(cond), Braced(braces), falseBraces(falseBraces) {
  /* pre-condition */
  BI_ASSERT(falseBraces);
}

inline biprog::Conditional::~Conditional() {
  //
}

#endif
