/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_TRANSFORM_HPP
#define BI_PROGRAM_TRANSFORM_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Transformation.
 *
 * @ingroup program
 */
class Transform: public virtual Named,
    public virtual Parenthesised,
    public virtual Braced,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Transform> {
public:
  /**
   * Constructor.
   */
  Transform(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Transform();

  virtual boost::shared_ptr<Expression> accept(Visitor& v);

    virtual bool operator<=(const Expression& o) const;
 virtual bool operator==(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Transform::~Transform() {
  //
}

#endif
