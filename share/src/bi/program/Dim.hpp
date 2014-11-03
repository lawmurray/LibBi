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
class Dim: public Named,
    public Bracketed,
    public boost::enable_shared_from_this<Dim> {
public:
  /**
   * Constructor.
   */
  Dim(const char* name, boost::shared_ptr<Expression> brackets);

  /**
   * Destructor.
   */
  virtual ~Dim();

  /*
   * Operators.
   */
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const Dim& o) const;
  virtual bool operator<=(const Dim& o) const;
  virtual bool operator>(const Dim& o) const;
  virtual bool operator>=(const Dim& o) const;
  virtual bool operator==(const Dim& o) const;
  virtual bool operator!=(const Dim& o) const;
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

inline bool biprog::Dim::operator<(const Dim& o) const {
  return *brackets < *o.brackets;
}

inline bool biprog::Dim::operator<=(const Dim& o) const {
  return *brackets <= *o.brackets;
}

inline bool biprog::Dim::operator>(const Dim& o) const {
  return *brackets > *o.brackets;
}

inline bool biprog::Dim::operator>=(const Dim& o) const {
  return *brackets >= *o.brackets;
}

inline bool biprog::Dim::operator==(const Dim& o) const {
  return *brackets == *o.brackets;
}

inline bool biprog::Dim::operator!=(const Dim& o) const {
  return *brackets != *o.brackets;
}

#endif
