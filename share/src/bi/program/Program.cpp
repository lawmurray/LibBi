/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Program.hpp"

#include "Named.hpp"
#include "EmptyExpression.hpp"
#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

biprog::Program::Program() :
    root(boost::make_shared<EmptyExpression>()) {
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

boost::shared_ptr<biprog::Typed> biprog::Program::getRoot() {
  return root;
}

void biprog::Program::setRoot(boost::shared_ptr<biprog::Typed> root) {
  this->root = root;
}

void biprog::Program::add(boost::shared_ptr<biprog::Typed> decl) {
  boost::shared_ptr<Named> named = boost::dynamic_pointer_cast < Named
      > (decl);
  if (named) {
    top()->add(named);
  }
}

boost::shared_ptr<biprog::Typed> biprog::Program::lookup(
    const char* name) {
//  BOOST_AUTO(iter, scopes.begin());
//  while (iter != scopes.end()) {
//    BOOST_AUTO(find, (*iter)->find(name));
//    if (find) {
//      return find;
//    }
//    ++iter;
//  }
  return boost::make_shared<EmptyExpression>();
}
