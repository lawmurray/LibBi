/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_GROUP_HPP
#define BI_PROGRAM_GROUP_HPP

#include "Expression.hpp"
#include "Delimiter.hpp"

namespace biprog {
/**
 * Group.
 *
 * @ingroup program
 */
class Group: public virtual Expression, public virtual boost::enable_shared_from_this<
    Group> {
public:
  /**
   * Constructor.
   */
  Group(const Delimiter delim, boost::shared_ptr<Expression> expr);

  /**
   * Destructor.
   */
  virtual ~Group();

  virtual boost::shared_ptr<Expression> accept(Visitor& v);

    virtual bool operator<=(const Expression& o) const;
 virtual bool operator==(const Expression& o) const;

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
  boost::shared_ptr<Expression> expr;
};
}

inline biprog::Group::Group(const Delimiter delim,
    boost::shared_ptr<Expression> expr) :
    delim(delim), expr(expr) {
  //
}

inline biprog::Group::~Group() {
  //
}

#endif
