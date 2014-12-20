/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_VISITOR_CPPREFERENCE_HPP
#define BI_VISITOR_CPPREFERENCE_HPP

#include "Visitor.hpp"

namespace biprog {
/**
 * Visitor for generating C++ code.
 */
class CppGenerator : public Visitor {
public:
  /**
   * Destructor.
   */
  virtual ~CppGenerator();
};
}

#endif
