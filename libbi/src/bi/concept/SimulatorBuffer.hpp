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
 * %SimulatorBuffer concept.
 *
 * @ingroup concept
 */
struct SimulatorBuffer {
  /**
   * Number of trajectory records.
   */
  int size1() const;

  /**
   * Number of time records.
   */
  int size2() const;

  /**
   * Read time.
   *
   * @param t Index of record.
   * @param[out] x Time.
   */
  void readTime(const int t, real& x);

  /**
   * Write time.
   *
   * @param t Index of record.
   * @param x Time.
   */
  void writeTime(const int t, const real& x);

  /**
   * Read times.
   *
   * @tparam V1 Vector type.
   *
   * @param t Index of first record.
   * @param T Number of records.
   * @param[out] x Times.
   */
  template<class V1>
  void readTimes(const int t, const int T, V1 x);

  /**
   * Write times.
   *
   * @tparam V1 Vector type.
   *
   * @param t Index of first record.
   * @param T Number of records.
   * @param x Times.
   */
  template<class V1>
  void writeTimes(const int t, const int T, const V1 x);

  /**
   * Read state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param t Time index.
   * @param[out] s State. Rows index trajectories, columns variables of the
   * given type.
   */
  template<class M1>
  void readState(const VarType type, const int t, M1& s);

  /**
   * Write state.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param t Time index.
   * @param s State. Rows index trajectories, columns variables of the given
   * type.
   */
  template<class M1>
  void writeState(const VarType type, const int t, const M1& s);

  /**
   * Read trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param[out] x Trajectory. Rows index variables of the given type,
   * columns times.
   */
  template<class M1>
  void readTrajectory(const VarType type, const int p, M1& x);

  /**
   * Write trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param[out] x Trajectory. Rows index variables of the given type,
   * columns times.
   */
  template<class M1>
  void writeTrajectory(const VarType type, const int p, const M1& x);

  /**
   * Read state of particular trajectory at particular time.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param t Time index.
   * @param[out] x State.
   */
  template<class V1>
  void readSingle(const VarType type, const int p, const int t, V1& x);

  /**
   * Write state of particular trajectory at particular time.
   *
   * @param type Node type.
   * @param p Trajectory index.
   * @param t Time index.
   * @param x State.
   */
  template<class V1>
  void writeSingle(const VarType type, const int p, const int t,
      const V1& x);
};

}
