/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CLASS_HPP
#define BI_PROGRAM_CLASS_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Derived.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Class.
 *
 * @ingroup program
 */
class Class: public virtual Named,
    public virtual Parenthesised,
    public virtual Derived,
    public virtual Braced,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Class> {
public:
  /**
   * Constructor.
   */
  Class(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> base,
      boost::shared_ptr<Expression> braces, boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Class();

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

inline biprog::Class::~Class() {
  //
}

#endif
