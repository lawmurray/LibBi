/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_SIMULATORNETCDFBUFFER_HPP
#define BI_BUFFER_SIMULATORNETCDFBUFFER_HPP

#include "NetCDFBuffer.hpp"
#include "../state/State.hpp"
#include "../math/locatable.hpp"
#include "../misc/Pipelineable.hpp"
#include "../math/temp_matrix.hpp"
#include "../method/misc.hpp"

#include <vector>

namespace bi {
/**
 * NetCDF buffer for storing, reading and writing results of Simulator.
 *
 * @ingroup io
 *
 * @section Concepts
 *
 * #concept::SimulatorBuffer
 */
class SimulatorNetCDFBuffer :
    public NetCDFBuffer,
    public Pipelineable<host_matrix_temp_type<real>::type> {
public:
  /**
   * Constructor.
   *
   * @param m BayesNet.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes and s-nodes should be
   * read/written.
   */
  SimulatorNetCDFBuffer(const BayesNet& m, const std::string& file,
      const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Constructor.
   *
   * @param m BayesNet.
   * @param P Number of trajectories to hold in file.
   * @param T Number of time points to hold in file.
   * @param file NetCDF file name.
   * @param mode File open mode.
   * @param flag Indicates whether or not p-nodes and s-nodes should be
   * read/written.
   */
  SimulatorNetCDFBuffer(const BayesNet& m, const int P, const int T,
      const std::string& file, const FileMode mode = READ_ONLY,
      const StaticHandling flag = STATIC_SHARED);

  /**
   * Destructor.
   */
  virtual ~SimulatorNetCDFBuffer();

  /**
   * @copydoc concept::SimulatorBuffer::size1()
   */
  int size1() const;

  /**
   * @copydoc concept::SimulatorBuffer::size2()
   */
  int size2() const;

  /**
   * @copydoc concept::SimulatorBuffer::readTime()
   */
  void readTime(const int t, real& x);

  /**
   * @copydoc concept::SimulatorBuffer::writeTime()
   */
  void writeTime(const int t, const real& x);

  /**
   * @copydoc concept::SimulatorBuffer::readState()
   */
  template<class M1>
  void readState(const NodeType type, const int t, M1 s);

  /**
   * @copydoc concept::SimulatorBuffer::writeState()
   */
  template<class M1>
  void writeState(const NodeType type, const int t, const M1 s,
      const int p = 0);

  /**
   * @copydoc concept::SimulatorBuffer::readTrajectory()
   */
  template<class M1>
  void readTrajectory(const NodeType type, const int p, M1 x);

  /**
   * @copydoc concept::SimulatorBuffer::writeTrajectory()
   */
  template<class M1>
  void writeTrajectory(const NodeType type, const int p, const M1 x);

  /**
   * @copydoc concept::SimulatorBuffer::readSingle()
   */
  template<class V1>
  void readSingle(const NodeType type, const int p, const int t, V1 x);

  /**
   * @copydoc concept::SimulatorBuffer::writeSingle()
   */
  template<class V1>
  void writeSingle(const NodeType type, const int p, const int t,
      const V1 x);

protected:
  /**
   * Set up structure of NetCDF file.
   *
   * @param P Number of particles.
   * @param T Number of time points.
   */
  void create(const long P, const long T);

  /**
   * Map structure of existing NetCDF file.
   *
   * @param P Number of particles. Used to validate file, ignored if
   * negative.
   * @param T Number of time points. Used to validate file, ignored if
   * negative.
   */
  void map(const long P = -1, const long T = -1);

  /**
   * Model.
   */
  const BayesNet& m;

  /**
   * Time dimension.
   */
  NcDim* nrDim;

  /**
   * Z-dimension.
   */
  NcDim* nzDim;

  /**
   * Y-dimension.
   */
  NcDim* nyDim;

  /**
   * X-dimension.
   */
  NcDim* nxDim;

  /**
   * P-dimension (trajectories).
   */
  NcDim* npDim;

  /**
   * Time variable.
   */
  NcVar* tVar;

  /**
   * Node variables, indexed by type.
   */
  std::vector<std::vector<NcVar*> > vars;

  /**
   * Static handling flag.
   */
  StaticHandling flag;
};
}

#include "../math/view.hpp"
#include "../math/temp_matrix.hpp"

inline int bi::SimulatorNetCDFBuffer::size1() const {
  return npDim->size();
}

inline int bi::SimulatorNetCDFBuffer::size2() const {
  return nrDim->size();
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readState(const NodeType type,
    const int t, M1 s) {
  long offsets[5], counts[5];
  BI_UNUSED NcBool ret;

  int start, id, j, size;
  for (id = 0; id < m.getNumNodes(type); ++id) {
    j = 0;
    size = 1;

    if (vars[type][id]->get_dim(j) == nrDim) {
      offsets[j] = t;
      counts[j] = 1;
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nzDim) {
      offsets[j] = 0;
      counts[j] = nzDim->size();
      size *= nzDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nyDim) {
      offsets[j] = 0;
      counts[j] = nyDim->size();
      size *= nyDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nxDim) {
      offsets[j] = 0;
      counts[j] = nxDim->size();
      size *= nxDim->size();
      ++j;
    }
    start = m.getNodeStart(type, id);

    counts[j] = s.size1();
    ret = vars[type][id]->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << vars[type][id]->name());

    if (M1::on_device) {
      clean();
      BOOST_AUTO(buf, host_temp_matrix<real>(s.size1(), s.size2()));
      ret = vars[type][id]->get(subrange(*buf, 0, buf->size1(), start, size).buf(), counts);
      s = *buf;
      add(buf);
    } else {
      ret = vars[type][id]->get(subrange(s, 0, s.size1(), start, size).buf(), counts);
    }
    BI_ASSERT(ret, "Inconvertible type reading " << vars[type][id]->name());
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeState(const NodeType type,
    const int t, const M1 s, const int p) {
  long offsets[5], counts[5];
  BI_UNUSED NcBool ret;

  int start, id, j, size;
  for (id = 0; id < m.getNumNodes(type); ++id) {
    j = 0;
    size = 1;

    if (vars[type][id]->get_dim(j) == nrDim) {
      offsets[j] = t;
      counts[j] = 1;
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nzDim) {
      offsets[j] = 0;
      counts[j] = nzDim->size();
      size *= nzDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nyDim) {
      offsets[j] = 0;
      counts[j] = nyDim->size();
      size *= nyDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nxDim) {
      offsets[j] = 0;
      counts[j] = nxDim->size();
      size *= nxDim->size();
      ++j;
    }
    start = m.getNodeStart(type, id);

    offsets[j] = p;
    counts[j] = s.size1();
    ret = vars[type][id]->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size writing " << vars[type][id]->name());

    if (M1::on_device || s.size1() != s.lead()) { // on device or not contiguous
      /* copy to contiguous buffer on host */
      BOOST_AUTO(buf, host_temp_matrix<real>(s.size1(), size));
      *buf = subrange(s, 0, s.size1(), start, size);
      synchronize();
      ret = vars[type][id]->put(buf->buf(), counts);
      delete buf;
    } else {
      ret = vars[type][id]->put(subrange(s, 0, s.size1(), start, size).buf(), counts);
    }
    BI_ASSERT(ret, "Inconvertible type writing " << vars[type][id]->name());
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::readTrajectory(const NodeType type,
    const int p, M1 x) {
  /* pre-conditions */
  assert (p < npDim->size());
  assert (x.size1() == m.getNetSize(type));
  assert (x.size2() == nrDim->size());

  int id, j, start, size;
  long offsets[5];
  long counts[5];
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumNodes(type); ++id) {
    start = m.getNodeStart(type, id);
    size = m.getNodeSize(type, id);
    j = 0;

    clean();
    BOOST_AUTO(x1, host_temp_matrix<typename M1::value_type>(size, nrDim->size()));
    assert(x1->size1() == x1->lead());

    if (vars[type][id]->get_dim(j) == nrDim) {
      counts[j] = nrDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nzDim) {
      counts[j] = nzDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nyDim) {
      counts[j] = nyDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nxDim) {
      counts[j] = nxDim->size();
      ++j;
    }
    offsets[j] = p;
    counts[j] = 1;

    ret = vars[type][id]->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << vars[type][id]->name());
    ret = vars[type][id]->get(x1->buf(), counts);
    BI_ASSERT(ret, "Inconvertible type reading " << vars[type][id]->name());
    rows(x, start, size) = *x1;
    add(x1);
  }
}

template<class M1>
void bi::SimulatorNetCDFBuffer::writeTrajectory(const NodeType type,
    const int p, const M1 x) {
  /* pre-conditions */
  assert (p < npDim->size());
  assert (x.size1() == m.getNetSize(type));
  assert (x.size2() == nrDim->size());

  int id, j, start, size;
  long offsets[5] = { 0, 0, 0, 0, 0 };
  long counts[5];
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumNodes(type); ++id) {
    start = m.getNodeStart(type, id);
    size = m.getNodeSize(type, id);
    j = 0;

    BOOST_AUTO(x1, host_temp_matrix<typename M1::value_type>(size, nrDim->size()));
    assert(x1->size1() == x1->lead());

    if (vars[type][id]->get_dim(j) == nrDim) {
      counts[j] = nrDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nzDim) {
      counts[j] = nzDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nyDim) {
      counts[j] = nyDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nxDim) {
      counts[j] = nxDim->size();
      ++j;
    }
    offsets[j] = p;
    counts[j] = 1;

    *x1 = rows(x, start, size);
    ret = vars[type][id]->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size writing " << vars[type][id]->name());
    synchronize();
    ret = vars[type][id]->put(x1->buf(), counts);
    BI_ASSERT(ret, "Inconvertible type writing " << vars[type][id]->name());
    delete x1;
  }
}

template<class V1>
void bi::SimulatorNetCDFBuffer::readSingle(const NodeType type,
    const int p, const int t, V1 x) {
  /* pre-conditions */
  assert (t >= 0 && t < nrDim->size());
  assert (p >= 0 && p < npDim->size());
  assert (x.size() == m.getNetSize(type));

  int id, j, start, size;
  long offsets[5];
  long counts[5];
  BI_UNUSED NcBool ret;

  for (id = 0; id < m.getNumNodes(type); ++id) {
    start = m.getNodeStart(type, id);
    size = 1;
    j = 0;

    if (vars[type][id]->get_dim(j) == nrDim) {
      offsets[j] = t;
      counts[j] = 1;
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nzDim) {
      counts[j] = nzDim->size();
      size *= nzDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nyDim) {
      counts[j] = nyDim->size();
      size *= nyDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nxDim) {
      counts[j] = nxDim->size();
      size *= nxDim->size();
      ++j;
    }
    offsets[j] = p;
    counts[j] = 1;

    ret = vars[type][id]->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size reading " << vars[type][id]->name());

    if (V1::on_device || x.inc() > 1) {
      clean();
      BOOST_AUTO(buf, host_temp_matrix<real>(size, 1));
      ret = vars[type][id]->get(buf->buf(), counts);
      subrange(x, start, size) = column(*buf, 0);
      add(buf);
    } else {
      ret = vars[type][id]->get(x.buf() + start, counts);
    }
    BI_ASSERT(ret, "Inconvertible type reading " << vars[type][id]->name());
  }
}

template<class V1>
void bi::SimulatorNetCDFBuffer::writeSingle(const NodeType type,
    const int p, const int t, const V1 x) {
  /* pre-conditions */
  assert (t >= 0 && t < nrDim->size());
  assert (p >= 0 && p < npDim->size());
  assert (x.size() == m.getNetSize(type) && x.inc() == 1);

  int id, j, start;
  long offsets[5];
  long counts[5];
  BI_UNUSED NcBool ret;

  BOOST_AUTO(buf, host_map_matrix(x));
  if (V1::on_device) {
    synchronize();
  }

  for (id = 0; id < m.getNumNodes(type); ++id) {
    start = m.getNodeStart(type, id);
    j = 0;

    if (vars[type][id]->get_dim(j) == nrDim) {
      offsets[j] = t;
      counts[j] = 1;
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nzDim) {
      counts[j] = nzDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nyDim) {
      counts[j] = nyDim->size();
      ++j;
    }
    if (vars[type][id]->get_dim(j) == nxDim) {
      counts[j] = nxDim->size();
      ++j;
    }
    offsets[j] = p;
    counts[j] = 1;

    ret = vars[type][id]->set_cur(offsets);
    BI_ASSERT(ret, "Index exceeds size writing " << vars[type][id]->name());
    ret = vars[type][id]->put(buf->buf() + start, counts);
    BI_ASSERT(ret, "Inconvertible type writing " << vars[type][id]->name());
  }

  delete buf;
}

#endif
