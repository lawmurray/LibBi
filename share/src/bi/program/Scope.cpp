/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#include "Scope.hpp"

#include "MethodOverload.hpp"
#include "Method.hpp"
#include "FunctionOverload.hpp"
#include "Function.hpp"

void biprog::Scope::add(boost::shared_ptr<MethodOverload> overload) {
  boost::shared_ptr<Method> method;
  BOOST_AUTO(iter, decls.find(overload->name));
  if (iter != decls.end()) {
    method = boost::dynamic_pointer_cast<Method>(iter->second);
    if (method == NULL) {
      BI_ERROR_MSG(false,
          "non-method declaration '" << overload->name << "' already exists in same scope");
    }
  } else {
    method = boost::make_shared<Method>();
    decls.insert(std::make_pair(overload->name, method));
  }
  method->add(overload);
}

void biprog::Scope::add(boost::shared_ptr<FunctionOverload> overload) {
  boost::shared_ptr<Function> func;
  BOOST_AUTO(iter, decls.find(overload->name));
  if (iter != decls.end()) {
    func = boost::dynamic_pointer_cast<Function>(iter->second);
    if (func == NULL) {
      BI_ERROR_MSG(false,
          "non-function declaration '" << overload->name << "' already exists in same scope");
    }
  } else {
    func = boost::make_shared<Function>();
    decls.insert(std::make_pair(overload->name, func));
  }
  func->add(overload);
}

void biprog::Scope::add(boost::shared_ptr<Named> decl) {
  BOOST_AUTO(iter, decls.find(decl->name));
  if (iter != decls.end()) {
    BI_ERROR_MSG(false,
        "declaration '" << decl->name << "' already exists in same scope");
  } else {
    decls.insert(std::make_pair(decl->name, decl));
  }
}
