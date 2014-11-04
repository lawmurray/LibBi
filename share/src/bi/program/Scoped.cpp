/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Scoped.hpp"

#include "MethodOverload.hpp"
#include "Method.hpp"
#include "FunctionOverload.hpp"
#include "Function.hpp"

void biprog::Scoped::add(MethodOverload* overload) {
  Method* method;
  BOOST_AUTO(iter, decls.find(overload->name));
  if (iter != decls.end()) {
    method = dynamic_cast<Method*>(iter->second);
    if (method == NULL) {
      BI_ERROR_MSG(false,
          "non-method declaration '" << overload->name << "' already exists in same scope");
    }
  } else {
    method = new Method();
    decls.insert(std::make_pair(overload->name, method));
  }
  method->add(overload);
}

void biprog::Scoped::add(FunctionOverload* overload) {
  Function* func;
  BOOST_AUTO(iter, decls.find(overload->name));
  if (iter != decls.end()) {
    func = dynamic_cast<Function*>(iter->second);
    if (func == NULL) {
      BI_ERROR_MSG(false,
          "non-function declaration '" << overload->name << "' already exists in same scope");
    }
  } else {
    func = new Function();
    decls.insert(std::make_pair(overload->name, func));
  }
  func->add(overload);
}

void biprog::Scoped::add(Named* decl) {
  BOOST_AUTO(iter, decls.find(decl->name));
  if (iter != decls.end()) {
    BI_ERROR_MSG(false,
        "declaration '" << decl->name << "' already exists in same scope");
  } else {
    decls.insert(std::make_pair(decl->name, decl));
  }
}
