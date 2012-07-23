/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_MARKABLE_HPP
#define BI_MISC_MARKABLE_HPP

#include <stack>

namespace bi {
/**
 * Utility base for classes implementing the #concept::Markable concept.
 *
 * @ingroup misc
 *
 * @tparam T Item type.
 *
 * Items of type @p T are stored and restored via a FILO stack. Typically each
 * item holds the state of the object at some instant, presumably some
 * configuration of all mutable class attributes, that can be restored later.
 */
template<class T>
class Markable {
protected:
  /**
   * Store item.
   *
   * @param o Item.
   */
  void mark(const T& o);

  /**
   * Restore item.
   *
   * @param o[out] Item.
   */
  void restore(T& o);

  /**
   * Empty all items.
   */
  void unmark();

  /**
   * Get top item.
   *
   * @param o[out] Item.
   */
  void top(T& o);

  /**
   * Pop top item.
   */
  void pop();

private:
  /**
   * Saved states.
   */
  std::stack<T*> os;
};

}

template<class T>
inline void bi::Markable<T>::mark(const T& o) {
  os.push(new T(o));
}

template<class T>
inline void bi::Markable<T>::restore(T& o) {
  top(o);
  pop();
}

template<class T>
inline void bi::Markable<T>::unmark() {
  while (!os.empty()) {
    pop();
  }
}

template<class T>
inline void bi::Markable<T>::top(T& o) {
  /* pre-condition */
  assert (!os.empty());

  o = *os.top();
}

template<class T>
inline void bi::Markable<T>::pop() {
  /* pre-condition */
  assert (!os.empty());

  delete os.top();
  os.pop();
}

#endif
