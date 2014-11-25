/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DIM_HPP
#define BI_PROGRAM_DIM_HPP

#include "Named.hpp"
#include "Bracketed.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Dimension.
 *
 * @ingroup program
 */
class Dim: public virtual Named,
    public virtual Bracketed,
    public virtual Typed {
public:
  /**
   * Constructor.
   */
  Dim(const std::string name, Typed* brackets);

  /**
   * Destructor.
   */
  virtual ~Dim();

  virtual Typed* clone();
  virtual Typed* accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Dim::Dim(const std::string name, Typed* brackets) :
    Named(name), Bracketed(brackets) {
  //
}

inline biprog::Dim::~Dim() {
  //
}

#endif
