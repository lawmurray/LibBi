/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_VAR_HPP
#define BI_PROGRAM_VAR_HPP

#include "Named.hpp"
#include "Bracketed.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public virtual Named,
    public virtual Bracketed,
    public virtual Typed,
    public boost::enable_shared_from_this<Var> {
public:
  /**
   * Constructor.
   */
  Var(const char* name, boost::shared_ptr<Expression> brackets, Type* type);

  /**
   * Destructor.
   */
  virtual ~Var();

  /*
   * Operators.
   */
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Var::Var(const char* name,
    boost::shared_ptr<Expression> brackets, Type* type) :
    Named(name), Bracketed(brackets), Typed(type) {
  //
}

inline biprog::Var::~Var() {
  //
}

#endif
