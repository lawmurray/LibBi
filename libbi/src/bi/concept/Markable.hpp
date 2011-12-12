/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

namespace concept {
/**
 * %Markable concept.
 *
 * @ingroup concept
 *
 * @note This is a phony class, representing a concept, for documentation
 * purposes only.
 */
struct Markable {
  /**
   * Store the current state of the object.
   */
  void mark() = 0;

  /**
   * Restore a previously marked state of the object. States are restored via
   * a FIFO queue according to the order in which they were marked.
   */
  void restore() = 0;
};
}
