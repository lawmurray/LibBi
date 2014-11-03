/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_NAMED_HPP
#define BI_PROGRAM_NAMED_HPP

#include "Expression.hpp"

#include <string>

namespace biprog {
/**
 * Named object.
 *
 * @ingroup program
 */
class Named : public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   */
  Named(const char* name);

  /**
   * Destructor.
   */
  virtual ~Named() = 0;

  /**
   * Name.
   */
  std::string name;
};
}

inline biprog::Named::Named(const char* name) : name(name) {
  //
}

inline biprog::Named::~Named() {
  //
}

#endif
