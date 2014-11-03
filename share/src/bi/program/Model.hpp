/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_MODEL_HPP
#define BI_PROGRAM_MODEL_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Model.
 *
 * @ingroup program
 */
class Model: public Named,
    public Parenthesised,
    public Braced,
    public boost::enable_shared_from_this<Model> {
public:
  /**
   * Constructor.
   */
  Model(const char* name, boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Model();

  /*
   * Operators.
   */
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const Model& o) const;
  virtual bool operator<=(const Model& o) const;
  virtual bool operator>(const Model& o) const;
  virtual bool operator>=(const Model& o) const;
  virtual bool operator==(const Model& o) const;
  virtual bool operator!=(const Model& o) const;
};
}

inline biprog::Model::Model(const char* name,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Named(name), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::Model::~Model() {
  //
}

inline bool biprog::Model::operator<(const Model& o) const {
  return *parens < *o.parens && *braces < *o.braces;
}

inline bool biprog::Model::operator<=(const Model& o) const {
  return *parens <= *o.parens && *braces <= *o.braces;
}

inline bool biprog::Model::operator>(const Model& o) const {
  return *parens > *o.parens && *braces > *o.braces;
}

inline bool biprog::Model::operator>=(const Model& o) const {
  return *parens >= *o.parens && *braces >= *o.braces;
}

inline bool biprog::Model::operator==(const Model& o) const {
  return *parens == *o.parens && *braces == *o.braces;
}

inline bool biprog::Model::operator!=(const Model& o) const {
  return *parens != *o.parens || *braces != *o.braces;
}

#endif

