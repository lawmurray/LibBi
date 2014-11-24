/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Scoped.hpp"

#include "EmptyExpression.hpp"
#include "../exception/AmbiguousReferenceException.hpp"
#include "../exception/UnresolvedReferenceException.hpp"
#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

#include <list>

bool biprog::Scoped::resolve(boost::shared_ptr<Reference> ref)
    throw (AmbiguousReferenceException) {
  BOOST_AUTO(iter, decls.find(ref->name));
  if (iter != decls.end()) {
    std::list<pointer_type> matches;
    iter->second->find(ref, matches);
    if (matches.size() == 1) {
      return true;
    } else if (matches.size() > 1) {
      throw AmbiguousReferenceException(ref, matches);
    }
  }
  return false;
}

void biprog::Scoped::add(boost::shared_ptr<Named> decl) {
  BOOST_AUTO(key, decl->name);
  BOOST_AUTO(val, boost::make_shared<poset_type>());
  BOOST_AUTO(pair, std::make_pair(key, val));
  BOOST_AUTO(iter, decls.insert(pair).first);
  iter->second->insert(decl);
}
