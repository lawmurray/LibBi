/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

#include <vector>

namespace concept {
/**
 * %SparseInputBuffer concept.
 *
 * @ingroup concept
 */
struct SparseInputBuffer {
  /**
   * Read from file.
   *
   * @param[out] s State.
   *
   * Reads the next record, based on time, into @p s. Only those variables
   * updated at this next time are modified.
   */
  void read(State& s) = 0;

  /**
   * Reset to beginning of buffer.
   */
  void rewind() = 0;

  /**
   * Get current time.
   *
   * @return The current time.
   */
  real getTime() = 0;

  /**
   * Is there another record yet?
   *
   * @return True if there is another record, false if the end of the buffer
   * has been reached.
   */
  bool hasNext() = 0;

  /**
   * Get next time.
   *
   * @return The next time in the file.
   */
  real getNextTime() = 0;

  /**
   * Get number of nodes updated at the current time.
   *
   * @param type Node type.
   *
   * @return Number of nodes updated at the current time.
   */
  int countCurrentNodes(const VarType type);

  /**
   * Get number of nodes to be updated at the next time.
   *
   * @param type Node type.
   *
   * @return Number of nodes to be updated at the next time.
   */
  int countNextNodes(const VarType type);

  /**
   * Get nodes that were updated at the current time.
   *
   * @param type Node type.
   * @param[out] ids Ids of nodes of the given type updated at current time.
   * Will be resized to fit.
   */
  template<class V1>
  void getCurrentNodes(const VarType type, V1& ids);

  /**
   * Get ids of nodes to be updated at the next time.
   *
   * @param type Node type.
   * @param[out] ids Ids of nodes of the given type to be updated at next
   * time. Will be resized to fit.
   */
  template<class V1>
  void getNextNodes(const VarType type, V1& ids);

  /**
   * Calculate number of unique time points in file.
   *
   * @param T Maximum time.
   *
   * @return Number of unique time points in file up to and including time
   * @p T. @p T is itself considered one of these times.
   *
   * This is particularly useful for determining the number of records to
   * reserve in output buffers, for example.
   */
  int countUniqueTimes(const real T) = 0;
};
}
