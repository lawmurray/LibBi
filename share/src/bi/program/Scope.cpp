/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Scope.hpp"

#include "Method.hpp"
#include "Method.hpp"
#include "Function.hpp"
#include "Function.hpp"
#include "EmptyExpression.hpp"
#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

//boost::shared_ptr<biprog::Expression> biprog::Scope::find(const char* name) {
//  BOOST_AUTO(iter, decls.find(name));
//  if (iter != decls.end()) {
//    return iter->find;
//  } else {
//    return boost::make_shared<EmptyExpression>();
//  }
//}

void biprog::Scope::add(boost::shared_ptr<Named> decl) {
  BOOST_AUTO(key, decl->name);
  BOOST_AUTO(val, boost::make_shared<poset_type>());
  BOOST_AUTO(pair, std::make_pair(key, val));
  BOOST_AUTO(iter, decls.insert(pair).first);
  iter->second->insert(decl);
}
