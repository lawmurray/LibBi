/**
 * @file
 *
 * Provides convenience methods for working with kernel density estimates.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/kde.hpp
 */
#ifndef BI_KD_KDE_HPP
#define BI_KD_KDE_HPP

#include "KDTree.hpp"

namespace bi {
/**
 * Calculate \f$h_{opt}\f$.
 *
 * @ingroup kd
 *
 * @param N Number of dimensions.
 * @param P Number of samples.
 *
 * Note:
 *
 * \f[
 * h_{opt} = \left[\frac{4}{(N+2)P}\right]^{\frac{1}{N+4}}\,,
 * \f]
 *
 * this being the optimal bandwidth for a kernel density estimate
 * over \f$P\f$ samples from a standard \f$N\f$-dimensional Gaussian
 * distribution, and Gaussian kernel (@ref Silverman1986
 * "Silverman, 1986"). We find this useful as a rule of thumb for
 * setting kernel density estimate bandwidths.
 */
double hopt(const int N, const int P);

/**
 * Dual-tree kernel density evaluation.
 *
 * @ingroup kd
 *
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
 * @tparam V2 Vector type.
 * @tparam M2 Matrix type.
 * @tparam K1 Kernel type.
 * @tparam V3 Vector type.
 *
 * @param queryTree Query tree.
 * @param targetTree Target tree.
 * @param K Kernel.
 * @param[out] p Vector of the density estimates for each of the points in
 * @p queryTree.
 * @param clear Clear @p p before computations?
 */
template<class V1, class M1, class V2, class M2, class K1, class V3>
void dualTreeDensity(KDTree<V1,M1>& queryTree, KDTree<V2,M2>& targetTree,
    const K1& K, V3 p, const bool clear = true);

/**
 * Self-tree kernel density evaluation.
 *
 * @ingroup kd
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam K1 Kernel type.
 * @tparam V2 Vector type.
 *
 * @param tree Tree.
 * @param X Samples.
 * @param lw Log-weights.
 * @param N Norm.
 * @param K Kernel.
 * @param[out] p Vector of the density estimates for each of the points in
 * @p tree.
 */
//template<class M1, class V1, class K1, class V2>
//void selfTreeDensity(KDTree<V1>& tree, const M1 X, const V1 lw,
//    const K1& K, V2 p);
/**
 * Cross-tree kernel density evaluation with multiple mixture model
 * weights. Simultaneously performs kernel density estimation of two 
 * trees to each other.
 *
 * @ingroup kd
 *
 * @tparam M1 Matrix type.
 * @tparam V1 Vector type.
 * @tparam K1 Kernel type.
 * @tparam M2 Matrix type.
 *
 * @param tree1 First tree.
 * @param tree2 Second tree.
 * @param X1 Samples for first tree.
 * @param X2 Samples for second tree.
 * @param lw1 Log-weights for first tree as target.
 * @param lw2 Log-weights for second tree as target.
 * @param N Norm.
 * @param K Kernel.
 * @param P1 On return, unnormalised result for first tree as query,
 * second as target, is added to this.
 * @param P2 On return, unnormalised result for second tree as query,
 * first as target, is added to this.
 * @param clear Clear result matrices before addition.
 */
//template<class M1, class V1, class K1, class M2>
//void crossTreeDensities(KDTree<V1>& tree1, KDTree<V1>& tree2, const M1 X1,
//    const M1 X2, const V1 lw1, const V1 lw2, const K1& K, M2 P1, M2 P2,
//    const bool clear = true);
}

#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"
#include "../math/sim_temp_vector.hpp"
#include "../math/sim_temp_matrix.hpp"

#include <list>
#include <stack>

inline double bi::hopt(const int N, const int P) {
  return std::pow(4.0 / ((N + 2) * P), 1.0 / (N + 4));
}

template<class V1, class M1, class V2, class M2, class K1, class V3>
void bi::dualTreeDensity(KDTree<V1,M1>& queryTree, KDTree<V2,M2>& targetTree,
    const K1& K, V3 p, const bool clear) {
  typedef typename KDTree<V1,M1>::var_type query_var_type;
  typedef typename KDTree<V2,M2>::var_type target_var_type;

  BOOST_AUTO(queryRoot, queryTree.getRoot());
  BOOST_AUTO(targetRoot, targetTree.getRoot());
  if (clear) {
    p.clear();
  }
  if (queryRoot != NULL && targetRoot != NULL) {
#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
    omp_lock_t lock;
    omp_init_lock(&lock);
#endif

    /* start with breadth first search to build reasonable work set for
     * division between threads */
    std::list<const query_var_type*> queryNodes1;
    std::list<const target_var_type*> targetVars1;
    queryNodes1.push_back(queryRoot);
    targetVars1.push_back(targetRoot);

    sim_temp_vector<M1> x(queryTree.getSize());
    bool done = false;
#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
    while (!done && (int)queryNodes1.size() < 64*omp_get_max_threads()) {
#else
    while (!done) {
#endif
      BOOST_AUTO(queryNode, queryNodes1.front());
      BOOST_AUTO(targetVar, targetVars1.front());

      done = queryNode == NULL || !queryNode->isInternal()
          || targetVar == NULL || !targetVar->isInternal();
      if (!done) {
        targetVar->difference(*queryNode, *x);
        if (K(*x) > 0.0) {
          queryNodes1.push_back(queryNode->getLeft());
          targetVars1.push_back(targetVar->getLeft());

          queryNodes1.push_back(queryNode->getLeft());
          targetVars1.push_back(targetVar->getRight());

          queryNodes1.push_back(queryNode->getRight());
          targetVars1.push_back(targetVar->getLeft());

          queryNodes1.push_back(queryNode->getRight());
          targetVars1.push_back(targetVar->getRight());
        }
        queryNodes1.pop_front();
        targetVars1.pop_front();
        done = queryNodes1.empty();
      }
    }

    /* now multithread */
#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
    typename temp_host_matrix<real>::type P(p.size(), omp_get_max_threads());
#else
    typename temp_host_matrix<real>::type P(p.size(), 1);
#endif
    P.clear();

#pragma omp parallel
    {
#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();
#else
      int nthreads = 1;
      int tid = 0;
#endif
      typename V2::value_type q;
      int i, j;

#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
      omp_set_lock (&lock);
#endif
      sim_temp_vector<M1> x1(queryTree.getSize());
#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
      omp_unset_lock(&lock);
#endif
      BOOST_AUTO(x, x1.ref());

      /* take share of nodes */
      std::list<const query_var_type*> queryNodes;  // list or vector appears ~6% faster than stack
      std::list<const target_var_type*> targetVars;
      BOOST_AUTO(queryIter, queryNodes1.begin());
      BOOST_AUTO(targetIter, targetVars1.begin());
      i = 0;
      while (queryIter != queryNodes1.end()) {
        if ((i - tid) % nthreads == 0) {
          queryNodes.push_back(*queryIter);
          targetVars.push_back(*targetIter);
        }
        ++i;
        ++queryIter;
        ++targetIter;
      }

      /* traverse tree */
      while (!queryNodes.empty()) {
        BOOST_AUTO(queryNode, queryNodes.back());
        BOOST_AUTO(targetVar, targetVars.back());
        queryNodes.pop_back();
        targetVars.pop_back();

        if (queryNode->isInternal() || targetVar->isInternal()) {
          /* should we recurse? */
          targetVar->difference(*queryNode, x);
          if (K(x) > 0.0) {
            if (queryNode->isInternal()) {
              if (targetVar->isInternal()) {
                /* split both query and target nodes */
                queryNodes.push_back(queryNode->getLeft());
                targetVars.push_back(targetVar->getLeft());

                queryNodes.push_back(queryNode->getLeft());
                targetVars.push_back(targetVar->getRight());

                queryNodes.push_back(queryNode->getRight());
                targetVars.push_back(targetVar->getLeft());

                queryNodes.push_back(queryNode->getRight());
                targetVars.push_back(targetVar->getRight());
              } else {
                /* split query node only */
                queryNodes.push_back(queryNode->getLeft());
                targetVars.push_back(targetVar);

                queryNodes.push_back(queryNode->getRight());
                targetVars.push_back(targetVar);
              }
            } else {
              /* split target node only */
              queryNodes.push_back(queryNode);
              targetVars.push_back(targetVar->getLeft());

              queryNodes.push_back(queryNode);
              targetVars.push_back(targetVar->getRight());
            }
          }
        } else {
          if (queryNode->isLeaf() && targetVar->isLeaf()) {
            i = queryNode->getIndex();
            x = queryNode->getValue();
            axpy(-1.0, targetVar->getValue(), x);
            q = bi::exp(targetVar->getLogWeight() + K.logDensity(x));
            P(i, tid) += q;
          } else if (queryNode->isLeaf() && targetVar->isPrune()) {
            i = queryNode->getIndex();
            q = 0.0;
            for (j = 0; j < targetVar->getCount(); ++j) {
              x = queryNode->getValue();
              axpy(-1.0, column(targetVar->getValues(), j), x);
              q += bi::exp(targetVar->getLogWeights()(j) + K.logDensity(x));
            }
            P(i, tid) += q;
          } else if (queryNode->isPrune() && targetVar->isLeaf()) {
            const std::vector<int>& is = queryNode->getIndices();
            for (i = 0; i < (int)is.size(); ++i) {
              x = column(queryNode->getValues(), i);
              axpy(-1.0, targetVar->getValue(), x);
              q = bi::exp(targetVar->getLogWeight() + K.logDensity(x));
              P(is[i], tid) += q;
            }
          } else if (queryNode->isPrune() && targetVar->isPrune()) {
            const std::vector<int>& is = queryNode->getIndices();
            for (i = 0; i < queryNode->getCount(); ++i) {
              q = 0.0;
              for (j = 0; j < targetVar->getCount(); ++j) {
                x = column(queryNode->getValues(), i);
                axpy(-1.0, column(targetVar->getValues(), j), x);
                q += bi::exp(targetVar->getLogWeights()(j) + K.logDensity(x));
              }
              P(is[i], tid) += q;
            }
          }
        }
      }

#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
      omp_set_lock(&lock);
      omp_unset_lock(&lock);
#endif
    }
    sum_columns(P, p);
#if defined(ENABLE_OPENMP) and defined(HAVE_OMP_H)
    omp_destroy_lock (&lock);
#endif
  }
}

//template<class M1, class V1, class K1, class V2>
//void bi::selfTreeDensity(KDTree<V1>& tree, const M1 X, const V1 lw,
//    const K1& K, V2 p) {
//  /* pre-condition */
//  BI_ASSERT(lw.size() == X.size1());
//
//  typedef typename KDTree<V1>::var_type var_type;
//
//  BOOST_AUTO(root, tree.getRoot());
//  p.clear();
//  if (root != NULL) {
//    std::stack<var_type> queryNodes;
//    std::stack<var_type> targetVars;
//    std::stack<bool> doCrosses; // for query equals target tree optimisations
//
//    locatable_temp_vector<ON_HOST,real>::type x(X.size2());
//    int i, j;
//    bool doCross;
//    double d;
//
//    queryNodes.push(root);
//    targetVars.push(root);
//    doCrosses.push(false);
//
//    while (!queryNodes.empty()) {
//      BOOST_AUTO(queryNode, queryNodes.top());
//      BOOST_AUTO(targetVar, targetVars.top());
//      queryNodes.pop();
//      targetVars.pop();
//      doCross = doCrosses.top();
//      doCrosses.pop();
//
//      if (queryNode->isLeaf() && targetVar->isLeaf()) {
//        i = queryNode->getIndex();
//        j = targetVar->getIndex();
//        x = row(X, i);
//        axpy(-1.0, row(X, j), x);
//        d = K.logDensity(x);
//        p(i) += bi::exp(lw(j) + d);
//        if (doCross) {
//          p(j) += bi::exp(lw(i) + d);
//        }
//      } else if (queryNode->isLeaf() && targetVar->isPrune()) {
//        i = queryNode->getIndex();
//        const std::vector<int>& js = targetVar->getIndices();
//        for (j = 0; j < js.size(); j++) {
//          x = row(X, i);
//          axpy(-1.0, row(X, js[j]), x);
//          d = K.logDensity(x);
//          p(i) += bi::exp(lw(js[j]) + d);
//          if (doCross) {
//            p(js[j]) += bi::exp(lw(i) + d);
//          }
//        }
//      } else if (queryNode->isPrune() && targetVar->isLeaf()) {
//        const std::vector<int>& is = queryNode->getIndices();
//        j = targetVar->getIndex();
//        for (i = 0; i < is.size(); i++) {
//          x = row(X, is[i]);
//          axpy(-1.0, row(X, j), x);
//          d = K.logDensity(x);
//          p(is[i]) += bi::exp(lw(j) + d);
//          if (doCross) {
//            p(j) += bi::exp(lw(is[i]) + d);
//          }
//        }
//      } else if (queryNode->isPrune() && targetVar->isPrune()) {
//        const std::vector<int>& is = queryNode->getIndices();
//        const std::vector<int>& js = targetVar->getIndices();
//        for (i = 0; i < is.size(); i++) {
//          for (j = 0; j < js.size(); j++) {
//            x = row(X, is[i]);
//            axpy(-1.0, row(X, js[j]), x);
//            d = K.logDensity(x);
//            p(is[i]) += bi::exp(lw(js[j]) + d);
//            if (doCross) {
//              p(js[j]) += bi::exp(lw(is[i]) + d);
//            }
//          }
//        }
//      } else {
//        /* should we recurse? */
//        targetVar->difference(queryNode, x);
//        if (K(x) > 0.0) {
//          if (queryNode->isInternal()) {
//            if (targetVar->isInternal()) {
//              /* split both query and target nodes */
//              queryNodes.push(queryNode->getLeft());
//              targetVars.push(targetVar->getLeft());
//              doCrosses.push(doCross);
//
//              queryNodes.push(queryNode->getLeft());
//              targetVars.push(targetVar->getRight());
//              if (queryNode == targetVar) {
//                /* symmetric, so just double left-right evaluation */
//                doCrosses.push(true);
//              } else {
//                /* asymmetric, so evaluate right-left separately */
//                doCrosses.push(doCross);
//
//                queryNodes.push(queryNode->getRight());
//                targetVars.push(targetVar->getLeft());
//                doCrosses.push(doCross);
//              }
//
//              queryNodes.push(queryNode->getRight());
//              targetVars.push(targetVar->getRight());
//              doCrosses.push(doCross);
//            } else {
//              /* split query node only */
//              queryNodes.push(queryNode->getLeft());
//              targetVars.push(targetVar);
//              doCrosses.push(doCross);
//
//              queryNodes.push(queryNode->getRight());
//              targetVars.push(targetVar);
//              doCrosses.push(doCross);
//            }
//          } else {
//            /* split target node only */
//            queryNodes.push(queryNode);
//            targetVars.push(targetVar->getLeft());
//            doCrosses.push(doCross);
//
//            queryNodes.push(queryNode);
//            targetVars.push(targetVar->getRight());
//            doCrosses.push(doCross);
//          }
//        }
//      }
//    }
//  }
//}

//template<class M1, class V1, class K1, class V2>
//void bi::crossTreeDensities(KDTree<V1>& tree1, KDTree<V1>& tree2,
//    const M1 X1, const M1 X2, const V1 lw1, const V1 lw2, const K1& K, V2 p1,
//    V2 p2,  const bool clear) {
//  /* pre-condition */
//  BI_ASSERT(lw1.size() == X1.size1());
//  BI_ASSERT(lw2.size() == X2.size1());
//  BI_ASSERT(p1.size() == X1.size1());
//  BI_ASSERT(p2.size() == X2.size1());
//  BI_ASSERT(X1.size2() == X2.size2());
//
//  typedef typename KDTree<V1>::var_type var_type;
//
//  BOOST_AUTO(root1, tree1.getRoot());
//  BOOST_AUTO(root2, tree2.getRoot());
//  if (clear) {
//    p1.clear();
//    p2.clear();
//  }
//  if (root1 != NULL && root2 != NULL) {
//    std::stack<var_type> nodes1;
//    std::stack<var_type> nodes2;
//
//    locatable_temp_vector<ON_HOST,real>::type x(X1.size2());
//    int i, j;
//    double d;
//
//    nodes1.push(root1);
//    nodes2.push(root2);
//
//    while (!nodes1.empty()) {
//      BOOST_AUTO(node1, nodes1.top());
//      BOOST_AUTO(node2, nodes2.top());
//      nodes1.pop();
//      nodes2.pop();
//
//      if (node1->isLeaf() && node2->isLeaf()) {
//        i = node1->getIndex();
//        j = node2->getIndex();
//        x = row(X1, i);
//        axpy(-1.0, row(X2,j), x);
//        d = K.logDensity(x);
//        p1(i) += bi::exp(lw2(j) + d);
//        p2(j) += bi::exp(lw1(i) + d);
//      } else if (node1->isLeaf() && node2->isPrune()) {
//        i = node1->getIndex();
//        const std::vector<int>& js = node2->getIndices();
//        for (j = 0; j < js.size(); ++j) {
//          x = row(X1, i);
//          axpy(-1.0, row(X2, js[j]), x);
//          d = K.logDensity(x);
//          p1(i) += bi::exp(lw2(js[j]) + d);
//          p2(js[j]) += bi::exp(lw1(i) + d);
//        }
//      } else if (node1->isPrune() && node2->isLeaf()) {
//        const std::vector<int>& is = node1->getIndices();
//        j = node2->getIndex();
//        for (i = 0; i < is.size(); ++i) {
//          x = row(X1, is[i]);
//          axpy(-1.0, row(X2, j), x);
//          d = K.logDensity(x);
//          p1(is[i]) += bi::exp(lw2(j) + d);
//          p2(j) += bi::exp(lw1(is[i]) + d);
//        }
//      } else if (node1->isPrune() && node2->isPrune()) {
//        const std::vector<int>& is = node1->getIndices();
//        const std::vector<int>& js = node2->getIndices();
//        for (i = 0; i < is.size(); ++i) {
//          for (j = 0; j < js.size(); ++j) {
//            x = row(X1, is[i]);
//            axpy(-1.0, row(X2, js[j]), x);
//            d = K(x);
//            p1(is[i]) += bi::exp(lw2(js[j]) + d);
//            p2(js[j]) += bi::exp(lw1(is[i]) + d);
//          }
//        }
//      } else {
//        /* should we recurse? */
//        node2->difference(node1, x);
//        if (K(x) > 0.0) {
//          if (node1->isInternal()) {
//            if (node2->isInternal()) {
//              /* split both query and target nodes */
//              nodes1.push(node1->getLeft());
//              nodes2.push(node2->getLeft());
//
//              nodes1.push(node1->getLeft());
//              nodes2.push(node2->getRight());
//
//              nodes1.push(node1->getRight());
//              nodes2.push(node2->getLeft());
//
//              nodes1.push(node1->getRight());
//              nodes2.push(node2->getRight());
//            } else {
//              /* split query node only */
//              nodes1.push(node1->getLeft());
//              nodes2.push(node2);
//
//              nodes1.push(node1->getRight());
//              nodes2.push(node2);
//            }
//          } else {
//            /* split target node only */
//            nodes1.push(node1);
//            nodes2.push(node2->getLeft());
//
//            nodes1.push(node1);
//            nodes2.push(node2->getRight());
//          }
//        }
//      }
//    }
//  }
//}

#endif
