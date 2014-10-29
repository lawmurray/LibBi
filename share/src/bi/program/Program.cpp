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

biprog::Declaration* biprog::Program::add(biprog::Declaration* decl) {
  scopes.front()[decl->name] = decl;
  return decl;
}

biprog::Reference* biprog::Program::lookup(const char* name,
    boost::shared_ptr<biprog::Expression> brackets,
    boost::shared_ptr<biprog::Expression> parens,
    boost::shared_ptr<biprog::Expression> braces) {
  std::deque<biprog::Match> matches;
  biprog::Match match;

  BOOST_AUTO(scope, scopes.begin());
  bool found = false;
  while (!found && scope != scopes.end()) {
    BOOST_AUTO(decls, scope->equal_range(name));
    found = decls.first != decls.second;

    BOOST_AUTO(decl, decls.first);
    while (decl != decls.second) {
      if (decl->second->match(brackets, parens, braces, match)) {
        matches.push_back(match);
        match.clear();
      }
      ++decl;
    }
    ++scope;
  }

  if (matches.size() > 0) {
    std::sort(matches.begin(), matches.end());
  } else {
    //yywarn("no match for symbol");
  }

  return new biprog::Reference(name, brackets, parens, braces);
}
