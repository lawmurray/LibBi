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
#include "Expression.hpp"

namespace biprog {
/**
 * Type.
 *
 * @ingroup program
 */
class Type : public Named {
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
};
}

inline biprog::Type::Type(const char* name) : Named(name) {
  //
}

inline biprog::Type::~Type() {
  //
}

#endif
