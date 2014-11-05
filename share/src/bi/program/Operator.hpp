/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_OPERATOR_HPP
#define BI_PROGRAM_OPERATOR_HPP

#include <string>

namespace biprog {
/**
 * Operator codes.
 *
 * @ingroup program
 */
enum Operator {
  OP_TRAVERSE,
  OP_TYPE,
  OP_DEFAULT,
  OP_POS,
  OP_NEG,
  OP_NOT,
  OP_POW,
  OP_ELEM_POW,
  OP_MUL,
  OP_ELEM_MUL,
  OP_DIV,
  OP_ELEM_DIV,
  OP_MOD,
  OP_ADD,
  OP_SUB,
  OP_LEFT,
  OP_RIGHT,
  OP_LT,
  OP_GT,
  OP_LE,
  OP_GE,
  OP_EQ,        /// ==
  OP_NE,
  OP_BIT_AND,   /// bitwise and
  OP_BIT_XOR,   /// bitwise exclusive or
  OP_BIT_OR,    /// bitwise inclusive or
  OP_AND,       /// logical and
  OP_OR,        /// logical or
  OP_EQUALS,    /// =
  OP_SIMTO,     /// ~
  OP_LEFT_ARROW,
  OP_RIGHT_ARROW,
  OP_COMMA,
  OP_SEMICOLON
};

/**
 * Operator strings.
 *
 * @ingroup program
 */
extern const char* const ops[];

}

/**
 * Output an operator.
 */
std::ostream& operator<<(std::ostream& out, const biprog::Operator op);

#endif
