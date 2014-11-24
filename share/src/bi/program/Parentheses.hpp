/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PARENTHESES_HPP
#define BI_PROGRAM_PARENTHESES_HPP

#include "Typed.hpp"
#include "Grouped.hpp"

namespace biprog {
/**
 * Parentheses.
 *
 * @ingroup program
 */
class Parentheses: public virtual Typed,
    public virtual Grouped,
    public virtual boost::enable_shared_from_this<Parentheses> {
public:
  /**
   * Constructor.
   */
  Parentheses(boost::shared_ptr<Typed> expr);

  /**
   * Destructor.
   */
  virtual ~Parentheses();

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

inline biprog::Parentheses::Parentheses(boost::shared_ptr<Typed> expr) :
    Typed(expr->type), Grouped(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Parentheses::~Parentheses() {
  //
}

#endif
