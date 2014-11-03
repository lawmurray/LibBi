/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Program.hpp"

biprog::Program::Program() {
  push();
}

biprog::Program::~Program() {
  pop();
}

void biprog::Program::push() {
  scopes.push_front(scope_type());
}

void biprog::Program::pop() {
  scopes.pop_front();
}

biprog::Named* biprog::Program::add(biprog::Named* decl) {
  scopes.front()[decl->name].insert(decl);
  return decl;
}

biprog::Reference* biprog::Program::lookup(const char* name,
    boost::shared_ptr<biprog::Expression> brackets,
    boost::shared_ptr<biprog::Expression> parens,
    boost::shared_ptr<biprog::Expression> braces) {
  return new Reference(name, brackets, parens, braces);
}
