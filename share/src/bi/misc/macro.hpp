/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_MACRO_HPP
#define BI_MISC_MACRO_HPP

/**
 * @def MACRO_QUOTE
 *
 * @arg x
 *
 * Quotes macro argument as string
 */
#define MACRO_QUOTE(x) #x

/**
 * @def BI_PASSTHROUGH_CONSTRUCTORS
 *
 * @arg Derived Derived type.
 * @arg Base Base type.
 *
 * Expands to declarations and definitions of passthrough constructors for
 * derived classes that must pass all of their constructor arguments to the
 * base type constructor.
 */
#define BI_PASSTHROUGH_CONSTRUCTORS(Derived, Base) \
  /** Pass-through constructor. */ \
  Derived() : \
      Base() {} \
  \
  /** Pass-through constructor. */ \
  template<class T1> \
  Derived(T1& o1) : \
      Base(o1) {} \
  \
  /** Pass-through constructor. */ \
  template<class T1, class T2> \
  Derived(T1& o1, T2& o2) : \
      Base(o1, o2) {} \
  \
  /** Pass-through constructor. */ \
  template<class T1, class T2, class T3> \
  Derived(T1& o1, T2& o2, T3& o3) : \
      Base(o1, o2, o3) {} \
  \
  /** Pass-through constructor. */ \
  template<class T1, class T2, class T3, class T4> \
  Derived(T1& o1, T2& o2, T3& o3, T4& o4) : \
      Base(o1, o2, o3, o4) {} \
  \
  /** Pass-through constructor. */ \
  template<class T1, class T2, class T3, class T4, class T5> \
  Derived(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5) : \
      Base(o1, o2, o3, o4, o5) {} \
  \
  /** Pass-through constructor. */ \
  template<class T1, class T2, class T3, class T4, class T5, class T6> \
  Derived(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5, T6& o6) : \
      Base(o1, o2, o3, o4, o5, o6) {}

#endif
