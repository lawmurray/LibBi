/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_TYPE_HPP
#define BI_PROGRAM_TYPE_HPP

#include "Named.hpp"

namespace biprog {
/**
 * Type.
 *
 * @ingroup program
 */
class Type: public virtual Named, public virtual boost::enable_shared_from_this<Type> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   */
  Type(const char* name);

  /**
   * Destructor.
   */
  virtual ~Type();

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

inline biprog::Type::Type(const char* name) :
    Named(name) {
  //
}

inline biprog::Type::~Type() {
  //
}

#endif
