/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
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
class Typed {
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

  /**
   * Type.
   */
  boost::scoped_ptr<Type> type;
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
