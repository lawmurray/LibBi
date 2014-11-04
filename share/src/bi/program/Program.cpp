/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Program.hpp"

#include "MethodOverload.hpp"
#include "FunctionOverload.hpp"
#include "Named.hpp"
#include "../misc/assert.hpp"

biprog::Program::Program() {
  push();
}

biprog::Program::~Program() {
  pop();
}

boost::shared_ptr<biprog::Scope> biprog::Program::top() {
  /* pre-condition */
  BI_ASSERT(scopes.size() > 0);

  return scopes.front();
}

void biprog::Program::push() {
  scopes.push_front(boost::make_shared<Scope>());
}

void biprog::Program::pop() {
  /* pre-condition */
  BI_ASSERT(scopes.size() > 0);

  scopes.pop_front();
}

void biprog::Program::add(boost::shared_ptr<biprog::Expression> decl) {
  boost::shared_ptr<Named> named = boost::dynamic_pointer_cast<Named>(decl);
  boost::shared_ptr<MethodOverload> method = boost::dynamic_pointer_cast<MethodOverload>(decl);
  boost::shared_ptr<FunctionOverload> function = boost::dynamic_pointer_cast<FunctionOverload>(decl);

  if (method) {
    top()->add(method);
  } else if (function) {
    top()->add(function);
  } else if (named) {
    top()->add(named);
  } else {
    BI_ASSERT(false);
  }
}

biprog::Reference* biprog::Program::lookup(const char* name,
    boost::shared_ptr<biprog::Expression> brackets,
    boost::shared_ptr<biprog::Expression> parens,
    boost::shared_ptr<biprog::Expression> braces) {
  return new Reference(name, brackets, parens, braces);
}
