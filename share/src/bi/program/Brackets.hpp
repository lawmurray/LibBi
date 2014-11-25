/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACKETS_HPP
#define BI_PROGRAM_BRACKETS_HPP

#include "Typed.hpp"
#include "Grouped.hpp"

namespace biprog {
/**
 * Square brackets.
 *
 * @ingroup program
 */
class Brackets: public virtual Typed, public virtual Grouped {
public:
  /**
   * Constructor.
   */
  Brackets(Typed* expr);

  /**
   * Destructor.
   */
  virtual ~Brackets();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Brackets::Brackets(Typed* expr) :
    Typed(expr->type->clone()), Grouped(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Brackets::~Brackets() {
  //
}

#endif
