/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_OPERATOR_HPP
#define BI_PROGRAM_OPERATOR_HPP

#include <string>

namespace biprog {
/**
 * Operator.
 *
 * @ingroup program
 *
 * @todo Flyweight this or use enum.
 */
class Operator {
public:
  /**
   * Constructor.
   *
   * @param op Single-character operator.
   */
  Operator(const char op);

  /**
   * Constructor.
   *
   * @param op Multiple-character operator.
   */
  Operator(const char* op);

  /**
   * Destructor.
   */
  virtual ~Operator();

  /**
   * Operator.
   */
  std::string op;
};
}

inline biprog::Operator::Operator(const char op) : op(1, op) {
  //
}

inline biprog::Operator::Operator(const char* op) : op(op) {
  //
}

inline biprog::Operator::~Operator() {
  //
}

#endif
