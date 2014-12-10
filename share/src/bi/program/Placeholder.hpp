/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PLACEHOLDER_HPP
#define BI_PROGRAM_PLACEHOLDER_HPP

#include "Statement.hpp"
#include "Named.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Placeholder symbol in def signature.
 *
 * @ingroup program
 */
class Placeholder: public virtual Statement,
    public virtual Named,
    public virtual Typed {
public:
  /**
   * Constructor.
   */
  Placeholder(const std::string name, Statement* type);

  /**
   * Destructor.
   */
  virtual ~Placeholder();

  virtual Placeholder* clone();
  virtual Statement* acceptStatement(Visitor& v);

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Placeholder::Placeholder(const std::string name, Statement* type) :
    Named(name), Typed(type) {
  //
}

inline biprog::Placeholder::~Placeholder() {
  //
}

#endif
