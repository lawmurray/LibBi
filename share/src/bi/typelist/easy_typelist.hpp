/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_TYPELIST_EASY_TYPELIST_HPP
#define BI_TYPELIST_EASY_TYPELIST_HPP

#include "typelist.hpp"
#include "front.hpp"
#include "back.hpp"
#include "empty.hpp"
#include "size.hpp"
#include "pop_back.hpp"
#include "pop_front.hpp"
#include "push_back.hpp"
#include "push_back_list.hpp"
#include "push_front.hpp"
#include "push_front_list.hpp"

namespace bi {
/**
 * Object-oriented type list.
 *
 * @ingroup typelist
 *
 * @tparam T typelist.
 */
template<class T = empty_typelist>
class easy_typelist {
public:
  /**
   * Get the specification of the first node.
   *
   * @return Specification of the first node.
   */
  typedef typename front<T>::type front;

  /**
   * Get the specification of the last node.
   *
   * @return Specification of the last node.
   */
  typedef typename back<T>::type back;

  /**
   * Is the specification empty?
   *
   * @return True if the specification is empty, false otherwise.
   */
  static const bool empty = bi::empty<T>::value;

  /**
   * Get the size of the specification.
   *
   * @return Number of nodes specified.
   */
  static const int size = bi::size<T>::value;

  /**
   * Remove the type of the last node.
   */
  typedef easy_typelist<typename pop_back<T>::type> pop_back;

  /**
   * Remove the type of the first node.
   */
  typedef easy_typelist<typename pop_front<T>::type> pop_front;

  /**
   * Push single type onto back.
   *
   * @tparam X A type.
   * @tparam N Number of instances of this type.
   *
   * @return Updated type list.
   */
  template<class X, int N = 1>
  struct push_back {
    typedef easy_typelist<typename bi::push_back<T,X,N>::type> type;
  };

  /**
   * Push type list onto back.
   *
   * @tparam S An easy_typelist type.
   * @tparam N Number of instances of this type list.
   *
   * @return Updated type list.
   */
  template<class S, int N = 1>
  struct push_back_spec {
    typedef easy_typelist<typename bi::push_back_list<T,typename S::type,
        N>::type> type;
  };

  /**
   * Push single type onto front.
   *
   * @tparam X A type.
   * @tparam N Number of instances of this type.
   *
   * @return Updated type list.
   */
  template<class X, int N = 1>
  struct push_front {
    typedef easy_typelist<typename bi::push_front<T,X,N>::type> type;
  };

  /**
   * Push type list onto front.
   *
   * @tparam S A easy_typelist type.
   * @tparam N Number of instances of this type.
   *
   * @return Updated type list.
   */
  template<class S, int N = 1>
  struct push_front_spec {
    typedef easy_typelist<typename bi::push_front_list<T,typename S::type,
        N>::type> type;
  };

  /**
   * Underlying type list.
   */
  typedef T type;

};

/**
 * @internal
 *
 * Object-oriented empty type list.
 */
template<>
class easy_typelist<empty_typelist> {
public:
  /**
   * Is the specification empty?
   *
   * @return True if the specification is empty, false otherwise.
   */
  static const bool empty = bi::empty<empty_typelist>::value;

  /**
   * Get the size of the specification.
   *
   * @return Number of nodes specified.
   */
  static const int size = bi::size<empty_typelist>::value;

  /**
   * @internal
   *
   * Push single type onto back.
   *
   * @tparam X A type.
   * @tparam N Number of instances of this type.
   *
   * @return Updated type list.
   */
  template<class X, int N = 1>
  struct push_back {
    typedef easy_typelist<typename bi::push_back<empty_typelist,X,
        N>::type> type;
  };

  /**
   * @internal
   *
   * Push type list onto back.
   *
   * @tparam S A easy_typelist type.
   * @tparam N Number of instances of this type list.
   *
   * @return Updated type list.
   */
  template<class S, int N = 1>
  struct push_back_spec {
    typedef easy_typelist<typename bi::push_back_list<empty_typelist,
        typename S::type,N>::type> type;
  };

  /**
   * @internal
   *
   * Push single type onto front.
   *
   * @tparam X A type.
   * @tparam N Number of instances of this type.
   *
   * @return Updated type list.
   */
  template<class X, int N = 1>
  struct push_front {
    typedef easy_typelist<typename bi::push_front<empty_typelist,X,
        N>::type> type;
  };

  /**
   * @internal
   *
   * Push type list onto front.
   *
   * @tparam S A easy_typelist type.
   * @tparam N Number of instances of this type.
   *
   * @return Updated type list.
   */
  template<class S, int N = 1>
  struct push_front_spec {
    typedef easy_typelist<typename bi::push_front_list<
        empty_typelist,typename S::type,N>::type> type;
  };

  /**
   * Underlying typelist.
   */
  typedef empty_typelist type;

};

/**
 * Empty easy_typelist.
 */
typedef easy_typelist<> empty_easy_typelist;

}

#endif
