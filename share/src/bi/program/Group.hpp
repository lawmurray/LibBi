/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_GROUP_HPP
#define BI_PROGRAM_GROUP_HPP

#include "Typed.hpp"
#include "Delimiter.hpp"

namespace biprog {
/**
 * Group.
 *
 * @ingroup program
 */
class Group: public virtual Typed,
    public virtual boost::enable_shared_from_this<Group> {
public:
  /**
   * Constructor.
   */
  Group(const Delimiter delim, boost::shared_ptr<Typed> expr);

  /**
   * Destructor.
   */
  virtual ~Group();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

  /**
   * Delimiter around group.
   */
  Delimiter delim;

  /**
   * Grouped expression.
   */
  boost::shared_ptr<Typed> expr;
};
}

inline biprog::Group::Group(const Delimiter delim,
    boost::shared_ptr<Typed> expr) :
    Typed(expr->type), delim(delim), expr(expr) {
  /* pre-condition */
  BI_ASSERT(expr);
}

inline biprog::Group::~Group() {
  //
}

#endif
