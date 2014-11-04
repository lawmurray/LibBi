/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Program.hpp"

#include "../misc/assert.hpp"

biprog::Program::Program() {
  push(this);
}

biprog::Program::~Program() {
  pop();
}

biprog::Scoped* biprog::Program::top() {
  return scopes.front();
}

void biprog::Program::push(Scoped* scope) {
  scopes.push_front(scope);
}

void biprog::Program::pop() {
  /* pre-condition */
  BI_ASSERT(scopes.size() > 1);  // should never pop self

  scopes.pop_front();
}

biprog::MethodOverload* biprog::Program::add(
    biprog::MethodOverload* overload) {
  top()->add(overload);
  return overload;
}

biprog::FunctionOverload* biprog::Program::add(
    biprog::FunctionOverload* overload) {
  top()->add(overload);
  return overload;
}

biprog::Named* biprog::Program::add(biprog::Named* decl) {
  top()->add(decl);
  return decl;
}

biprog::Reference* biprog::Program::lookup(const char* name,
    boost::shared_ptr<biprog::Expression> brackets,
    boost::shared_ptr<biprog::Expression> parens,
    boost::shared_ptr<biprog::Expression> braces) {
  return new Reference(name, brackets, parens, braces);
}
