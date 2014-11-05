/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_TYPED_HPP
#define BI_PROGRAM_TYPED_HPP

#include "Type.hpp"

namespace biprog {
/**
 * Typed object.
 *
 * @ingroup program
 */
class Typed : public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param type type.
   */
  Typed(Type* type);

  /**
   * Destructor.
   */
  virtual ~Typed() = 0;

protected:
  /**
   * Type.
   */
  boost::shared_ptr<Type> type;
};
}

inline biprog::Typed::Typed(Type* type) :
    type(type) {
  //
}

inline biprog::Typed::~Typed() {
  //
}

#endif
