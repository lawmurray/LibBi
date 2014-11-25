/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DIM_HPP
#define BI_PROGRAM_DIM_HPP

#include "Statement.hpp"
#include "Named.hpp"
#include "Bracketed.hpp"

namespace biprog {
/**
 * Dimension.
 *
 * @ingroup program
 */
class Dim: public virtual Statement,
    public virtual Named,
    public virtual Bracketed {
public:
  /**
   * Constructor.
   */
  Dim(const std::string name, Expression* brackets);

  /**
   * Destructor.
   */
  virtual ~Dim();

  virtual Dim* clone();
  virtual Statement* accept(Visitor& v);

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Dim::Dim(const std::string name, Expression* brackets) :
    Named(name), Bracketed(brackets) {
  //
}

inline biprog::Dim::~Dim() {
  //
}

#endif
