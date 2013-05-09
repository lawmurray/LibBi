/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_NELDERMEADOPTIMISER_HPP
#define BI_METHOD_NELDERMEADOPTIMISER_HPP

#include "../state/Schedule.hpp"
#include "../state/State.hpp"
#include "../math/gsl.hpp"

#include <gsl/gsl_multimin.h>

namespace bi {
/**
 * @internal
 *
 * %State of NelderMeadOptimiser.
 */
struct NelderMeadOptimiserState {
  /**
   * Constructor.
   *
   * @param m Model.
   */
  NelderMeadOptimiserState(const int M);

  /**
   * Destructor.
   */
  ~NelderMeadOptimiserState();

  /**
   * State.
   */
  gsl_vector* x;

  /**
   * Step size.
   */
  gsl_vector* step;

  /**
   * Minimizer.
   */
  gsl_multimin_fminimizer* minimizer;

  /**
   * Size.
   */
  double size;
};
}

inline bi::NelderMeadOptimiserState::NelderMeadOptimiserState(const int M) {
  x = gsl_vector_alloc(M);
  step = gsl_vector_alloc(M);
  minimizer = gsl_multimin_fminimizer_alloc(
      gsl_multimin_fminimizer_nmsimplex2, M);
}

inline bi::NelderMeadOptimiserState::~NelderMeadOptimiserState() {
  gsl_vector_free(x);
  gsl_vector_free(step);
  gsl_multimin_fminimizer_free(minimizer);
}

namespace bi {
/**
 * Parameter structure passed to function to optimise.
 */
template<class B, Location L, class F>
struct NelderMeadOptimiserParams {
  B* m;
  Random* rng;
  State<B,L>* s;
  F* filter;
  ScheduleIterator first, last;
};

/**
 * Nelder-Mead simplex optimisation.
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam F #concept::Filter type.
 * @tparam IO1 Output type.
 */
template<class B, class F, class IO1>
class NelderMeadOptimiser {
public:
  /**
   * State type.
   */
  typedef NelderMeadOptimiserState state_type;

  /**
   * Constructor.
   *
   * @tparam IO1 #concept::OptimiserBuffer type.
   *
   * @param m Model.
   * @param filter Filter.
   * @param out Output.
   * @param mode Mode of operation.
   *
   * @see ParticleFilter
   */
  NelderMeadOptimiser(B& m, F* filter = NULL, IO1* out = NULL,
      const OptimiserMode mode = MAXIMUM_LIKELIHOOD);

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Get filter.
   *
   * @return Filter.
   */
  F* getFilter();

  /**
   * Set filter.
   *
   * @param filter Filter.
   */
  void setFilter(F* filter);

  /**
   * Get output.
   *
   * @return Output.
   */
  IO1* getOutput();

  /**
   * Set output.
   *
   * @param out Output buffer.
   */
  void setOutput(IO1* out);

  /**
   * Optimise.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param inInit Initialisation file.
   * @param simplexSizeRel Size of simplex relative to each dimension.
   * @param stopSteps Maximum number of steps to take.
   * @param stopSize Size for stopping criterion.
   *
   * Note that @p s should be initialised with a starting state.
   */
  template<Location L, class IO2>
  void optimise(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, State<B,L>& s, IO2* inInit = NULL,
      const real simplexSizeRel = 0.1, const int stopSteps = 100,
      const real stopSize = 1.0e-4);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise.
   *
   * @tparam L Location.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s State.
   * @param inInit Initialisation file.
   * @param simplexSizeRel Size of simplex relative to each dimension.
   */
  template<Location L, class IO2>
  void init(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, State<B,L>& s, IO2* inInit,
      const real simplexSizeRel = 0.1);

  /**
   * Perform one iteration step of optimiser.
   */
  void step();

  /**
   * Has optimiser converged?
   *
   * @param stopSize Size for stopping criterion.
   */
  bool hasConverged(const real stopSize = 1.0e-4);

  /**
   * Output current state.
   *
   * @tparam L Location.
   *
   * @param k Index in output file.
   * @param s State.
   */
  template<Location L>
  void output(const int k, const State<B,L>& s);

  /**
   * Report progress on stderr.
   *
   * @param k Number of steps taken.
   */
  void report(const int k);

  /**
   * Terminate.
   */
  void term();
  //@}

private:
  /**
   * Model.
   */
  B& m;

  /**
   * Filter.
   */
  F* filter;

  /**
   * Output.
   */
  IO1* out;

  /**
   * Optimisation mode.
   */
  OptimiserMode mode;

  /**
   * Current state.
   */
  NelderMeadOptimiserState state;

  /**
   * Cost function for maximum likelihood.
   */
  template<Location L>
  static double ml(const gsl_vector* x, void* params);

  /**
   * Cost function for maximum a posteriori.
   */
  template<Location L>
  static double map(const gsl_vector* x, void* params);
};

/**
 * Factory for creating NelderMeadOptimiser objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see NelderMeadOptimiser
 */
template<Location CL = ON_HOST>
struct NelderMeadOptimiserFactory {
  /**
   * Create Nelder-Mead optimiser.
   *
   * @return NelderMeadOptimiser object. Caller has ownership.
   *
   * @see NelderMeadOptimiser::NelderMeadOptimiser()
   */
  template<class B, class F, class IO1>
  static NelderMeadOptimiser<B,F,IO1>* create(B& m, F* filter = NULL,
      IO1* out = NULL, const OptimiserMode mode = MAXIMUM_LIKELIHOOD) {
    return new NelderMeadOptimiser<B,F,IO1>(m, filter, out, mode);
  }
};
}

#include "../resampler/Resampler.hpp"
#include "../math/misc.hpp"
#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../misc/exception.hpp"

template<class B, class F, class IO1>
bi::NelderMeadOptimiser<B,F,IO1>::NelderMeadOptimiser(B& m, F* filter,
    IO1* out, const OptimiserMode mode) :
    m(m), filter(filter), out(out), mode(mode), state(B::NP) {
  //
}

template<class B, class F, class IO1>
F* bi::NelderMeadOptimiser<B,F,IO1>::getFilter() {
  return filter;
}

template<class B, class F, class IO1>
void bi::NelderMeadOptimiser<B,F,IO1>::setFilter(F* filter) {
  this->filter = filter;
}

template<class B, class F, class IO1>
IO1* bi::NelderMeadOptimiser<B,F,IO1>::getOutput() {
  return out;
}

template<class B, class F, class IO1>
void bi::NelderMeadOptimiser<B,F,IO1>::setOutput(IO1* out) {
  this->out = out;
}

template<class B, class F, class IO1>
template<bi::Location L, class IO2>
void bi::NelderMeadOptimiser<B,F,IO1>::optimise(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit, const real simplexSizeRel, const int stopSteps,
    const real stopSize) {
  int k = 0;
  init(rng, first, last, s, inInit, simplexSizeRel);
  while (k < stopSteps && !hasConverged(stopSize)) {
    step();
    report(k);
    output(k, s);
    ++k;
  }
  term();
}

template<class B, class F, class IO1>
template<bi::Location L, class IO2>
void bi::NelderMeadOptimiser<B,F,IO1>::init(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, State<B,L>& s,
    IO2* inInit, const real simplexSizeRel) {
  /* first run of filter to get everything initialised properly */
  filter->filter(rng, first, last, s, inInit);

  /* initialise state vector */
  BOOST_AUTO(x, gsl_vector_reference(state.x));
  x = vec(s.get(P_VAR));
  gsl_vector_reference(state.step) = x;
  mulscal_elements(gsl_vector_reference(state.step), simplexSizeRel,
      gsl_vector_reference(state.step));

  /* parameters */
  NelderMeadOptimiserParams<B,L,F>* params = new NelderMeadOptimiserParams<B,
      L,F>();  ///@todo Leaks
  params->m = &m;
  params->rng = &rng;
  params->s = &s;
  params->filter = filter;
  params->first = first;
  params->last = last;

  /* function */
  gsl_multimin_function* f = new gsl_multimin_function();  ///@todo Leaks
  if (mode == MAXIMUM_A_POSTERIORI) {
    f->f = NelderMeadOptimiser<B,F,IO1>::template map<L>;
  } else {
    f->f = NelderMeadOptimiser<B,F,IO1>::template ml<L>;
  }
  f->n = B::NP;
  f->params = params;

  gsl_multimin_fminimizer_set(state.minimizer, f, state.x, state.step);
}

template<class B, class F, class IO1>
void bi::NelderMeadOptimiser<B,F,IO1>::step() {
  int status = gsl_multimin_fminimizer_iterate(state.minimizer);
  BI_ERROR_MSG(status == GSL_SUCCESS, "iteration failed");
}

template<class B, class F, class IO1>
bool bi::NelderMeadOptimiser<B,F,IO1>::hasConverged(const real stopSize) {
  state.size = gsl_multimin_fminimizer_size(state.minimizer);
  return gsl_multimin_test_size(state.size, stopSize) == GSL_SUCCESS;
}

template<class B, class F, class IO1>
template<bi::Location L>
void bi::NelderMeadOptimiser<B,F,IO1>::output(const int k,
    const State<B,L>& s) {
  if (out != NULL) {
    out->writeState(P_VAR, k, vec(s.get(P_VAR)));
    //out->writeState(D_VAR, k, row(s.get(D_VAR), 0));
    out->writeValue(k, -state.minimizer->fval);
    out->writeSize(k, state.size);
  }
}

template<class B, class F, class IO1>
void bi::NelderMeadOptimiser<B,F,IO1>::report(const int k) {
  std::cerr << k << ":\t";
  std::cerr << "value=" << -state.minimizer->fval;
  std::cerr << '\t';
  std::cerr << "size=" << state.size;
  std::cerr << std::endl;
}

template<class B, class F, class IO1>
void bi::NelderMeadOptimiser<B,F,IO1>::term() {
  //
}

template<class B, class F, class IO1>
template<bi::Location L>
double bi::NelderMeadOptimiser<B,F,IO1>::ml(const gsl_vector* x,
    void* params) {
  typedef NelderMeadOptimiserParams<B,L,F> param_type;
  param_type* p = reinterpret_cast<param_type*>(params);

  /* evaluate */
  try {
    real ll = p->filter->filter(*p->rng, p->first, p->last,
        gsl_vector_reference(x), *p->s);
    return -ll;
  } catch (CholeskyException e) {
    return GSL_NAN;
  } catch (ParticleFilterDegeneratedException e) {
    return GSL_NAN;
  }
}

template<class B, class F, class IO1>
template<bi::Location L>
double bi::NelderMeadOptimiser<B,F,IO1>::map(const gsl_vector* x,
    void* params) {
  typedef NelderMeadOptimiserParams<B,L,F> param_type;
  param_type* p = reinterpret_cast<param_type*>(params);

  int P = p->s->size();
  p->s->resize(1, true);

  /* initialise */
  vec(p->s->get(PY_VAR)) = gsl_vector_reference(x);
  real lp = p->m->parameterLogDensity(*p->s);
  p->s->resize(P, true);

  /* evaluate */
  if (bi::is_finite(lp)) {
    try {
      real ll = p->filter->filter(*p->rng, p->first, p->last,
          gsl_vector_reference(x), *p->s);
      return -(ll + lp);
    } catch (CholeskyException e) {
      return GSL_NAN;
    }
  } else {
    return GSL_NAN;
  }
}

#endif
