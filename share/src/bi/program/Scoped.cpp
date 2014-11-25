/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Scoped.hpp"

#include "boost/typeof/typeof.hpp"

#include <list>

bool biprog::Scoped::resolve(Reference* ref)
    throw (AmbiguousReferenceException) {
  BOOST_AUTO(iter, decls.find(ref->name));
  if (iter != decls.end()) {
    std::list<pointer_type> matches;
    iter->second.find(ref, matches);
    if (matches.size() == 1) {
      return true;
    } else if (matches.size() > 1) {
      throw AmbiguousReferenceException(ref, matches);
    }
  }
  return false;
}

void biprog::Scoped::add(Named* decl) {
  BOOST_AUTO(key, decl->name);
  BOOST_AUTO(val, poset_type());
  BOOST_AUTO(pair, std::make_pair(key, val));
  BOOST_AUTO(iter, decls.insert(pair).first);
  iter->second.insert(dynamic_cast<Statement*>(decl));
}
