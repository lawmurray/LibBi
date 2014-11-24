/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACES_HPP
#define BI_PROGRAM_BRACES_HPP

#include "Typed.hpp"
#include "Grouped.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Braces.
 *
 * @ingroup program
 */
class Braces: public virtual Typed,
    public virtual Grouped,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Braces> {
public:
  /**
   * Constructor.
   */
  Braces();

  /**
   * Constructor.
   */
  Braces(boost::shared_ptr<Typed> expr);

  /**
   * Destructor.
   */
  virtual ~Braces();

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

inline biprog::Braces::Braces() {
  //
}

inline biprog::Braces::Braces(boost::shared_ptr<Typed> expr) :
    Typed(expr->type), Grouped(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Braces::~Braces() {
  //
}

#endif
