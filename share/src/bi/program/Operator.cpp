/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Operator.hpp"

const char* const biprog::ops[] = {
    ".",
    ":",
    "?",
    "+",
    "-",
    "!",
    "**",
    ".**",
    "*",
    ".*",
    "/",
    "./",
    "%",
    "+",
    "-",
    "<<",
    ">>",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
    "&",
    "^",
    "|",
    "&&",
    "||",
    "=",
    "~",
    "<-",
    "->",
    ",",
    ";"
  };

std::ostream& operator<<(std::ostream& out, const biprog::Operator op) {
  out << biprog::ops[op];
  return out;
}
