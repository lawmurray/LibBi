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

namespace biprog {
/**
 * Dimension.
 *
 * @ingroup program
 */
class Dim: public virtual Named,
    public virtual Bracketed,
    public virtual boost::enable_shared_from_this<Dim> {
public:
  /**
   * Constructor.
   */
  Dim(const char* name, boost::shared_ptr<Expression> brackets);

  /**
   * Destructor.
   */
  virtual ~Dim();

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

inline biprog::Dim::Dim(const char* name,
    boost::shared_ptr<Expression> brackets) :
    Named(name), Bracketed(brackets) {
  //
}

inline biprog::Dim::~Dim() {
  //
}

#endif
