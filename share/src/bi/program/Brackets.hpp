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
class Brackets: public virtual Typed,
    public virtual Grouped,
    public virtual boost::enable_shared_from_this<Brackets> {
public:
  /**
   * Constructor.
   */
  Brackets(boost::shared_ptr<Typed> expr);

  /**
   * Destructor.
   */
  virtual ~Brackets();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Brackets::Brackets(boost::shared_ptr<Typed> expr) :
    Typed(expr->type), Grouped(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Brackets::~Brackets() {
  //
}

#endif
