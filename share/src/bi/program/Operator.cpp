/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Operator.hpp"

const char* const biprog::ops[] = {
    ".",
    ":",
    "?",
    " + ",
    " - ",
    "!",
    "**",
    ".**",
    "*",
    ".*",
    "/",
    "./",
    " % ",
    " + ",
    " - ",
    " << ",
    " >> ",
    " < ",
    " > ",
    " <= ",
    " >= ",
    " == ",
    " != ",
    " & ",
    " ^ ",
    " | ",
    " && ",
    " || ",
    " = ",
    " ~ ",
    " <- ",
    " -> ",
    ", ",
    ";\n"
  };

std::ostream& operator<<(std::ostream& out, const biprog::Operator op) {
  out << biprog::ops[op];
  return out;
}
