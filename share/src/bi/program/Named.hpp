/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_NAMED_HPP
#define BI_PROGRAM_NAMED_HPP

#include <string>

namespace biprog {
/**
 * Named object.
 *
 * @ingroup program
 */
class Named {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   */
  Named(std::string name);

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

inline biprog::Named::Named(std::string name) : name(name) {
  //
}

inline biprog::Named::~Named() {
  //
}

#endif
