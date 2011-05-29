/**
 * @file
 *
 * Provides convenience methods for working with kernel density estimates.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1359 $
 * $Date: 2011-03-31 16:58:20 +0800 (Thu, 31 Mar 2011) $
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
 * @tparam V2 Vector type.
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 * @tparam V3 Vector type.
 * @tparam K1 Kernel type.
 * @tparam V4 Vector type.
 *
 * @param queryTree Query tree.
 * @param targetTree Target tree.
 * @param queryX Query samples.
 * @param targetX Target samples.
 * @param lw Log-weights.
 * @param K Kernel.
 * @param[out] p Vector of the density estimates for each of the points in
 * @p queryTree.
 */
template<class V1, class V2, class M1, class M2, class V3, class K1, class V4>
void dualTreeDensity(KDTree<V1>& queryTree, KDTree<V2>& targetTree,
    const M1 queryX, const M2 targetX, const V3 lw, const K1& K, V4 p);

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
template<class M1, class V1, class K1, class V2>
void selfTreeDensity(KDTree<V1>& tree, const M1 X, const V1 lw,
    const K1& K, V2 p);

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
template<class M1, class V1, class K1, class M2>
void crossTreeDensities(KDTree<V1>& tree1, KDTree<V1>& tree2, const M1 X1,
    const M1 X2, const V1 lw1, const V1 lw2, const K1& K, M2 P1, M2 P2,
    const bool clear = true);

}

#include <list>
#include <stack>

#include "../math/locatable.hpp"

inline double bi::hopt(const int N, const int P) {
  return std::pow(4.0/((N + 2)*P), 1.0/(N + 4));
}

template<class V1, class V2, class M1, class M2, class V3, class K1, class V4>
void bi::dualTreeDensity(KDTree<V1>& queryTree, KDTree<V2>& targetTree,
    const M1 queryX, const M2 targetX, const V3 lw, const K1& K, V4 p) {
  /* pre-condition */
  assert (lw.size() == targetX.size1());
  assert (queryX.size2() == targetX.size2());
  assert (p.size() == queryX.size1());

  typedef KDTreeNode<V1> query_node_type;
  typedef KDTreeNode<V2> target_node_type;
  
  BOOST_AUTO(queryRoot, queryTree.getRoot());
  BOOST_AUTO(targetRoot, targetTree.getRoot());
  p.clear();
  if (queryRoot != NULL && targetRoot != NULL) {
    /* buffers */
    BOOST_AUTO(transQueryX1, temp_matrix<M1>(queryX.size2(), queryX.size1()));
    BOOST_AUTO(transTargetX1, temp_matrix<M2>(targetX.size2(), targetX.size1()));
    BOOST_AUTO(transQueryX, transQueryX1->ref());
    BOOST_AUTO(transTargetX, transTargetX1->ref());
    transpose(queryX, transQueryX);
    transpose(targetX, transTargetX);

    omp_lock_t lock;
    omp_init_lock(&lock);

    /* start with breadth first search to build reasonable work set for
     * division between threads */
    std::list<const query_node_type*> queryNodes1;
    std::list<const target_node_type*> targetNodes1;
    queryNodes1.push_back(queryRoot);
    targetNodes1.push_back(targetRoot);

    BOOST_AUTO(x, temp_vector<M1>(queryX.size2()));
    bool done = false;
    while (!done && (int)queryNodes1.size() < 16*omp_get_max_threads()) {
      BOOST_AUTO(queryNode, queryNodes1.front());
      BOOST_AUTO(targetNode, targetNodes1.front());

      done = !(queryNode != NULL && queryNode->isInternal() &&
          targetNode != NULL && targetNode->isInternal());
      if (!done) {
        targetNode->difference(*queryNode, *x);
        if (K(*x) > 0.0) {
          queryNodes1.push_back(queryNode->getLeft());
          targetNodes1.push_back(targetNode->getLeft());

          queryNodes1.push_back(queryNode->getLeft());
          targetNodes1.push_back(targetNode->getRight());

          queryNodes1.push_back(queryNode->getRight());
          targetNodes1.push_back(targetNode->getLeft());

          queryNodes1.push_back(queryNode->getRight());
          targetNodes1.push_back(targetNode->getRight());
        }
        queryNodes1.pop_front();
        targetNodes1.pop_front();
      }
    }
    delete x;

    /* now multithread */
    BOOST_AUTO(P, host_temp_matrix<real>(p.size(), omp_get_max_threads()));
    P->clear();

    #pragma omp parallel
    {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();
      typename V2::value_type q;
      int i, j, evals = 0;

      omp_set_lock(&lock);
      BOOST_AUTO(x1, temp_vector<M1>(queryX.size2()));
      omp_unset_lock(&lock);
      BOOST_AUTO(x, x1->ref());

      /* take share of nodes */
      std::list<const query_node_type*> queryNodes; // list or vector appears ~6% faster than stack
      std::list<const target_node_type*> targetNodes;
      BOOST_AUTO(queryIter, queryNodes1.begin());
      BOOST_AUTO(targetIter, targetNodes1.begin());
      i = 0;
      while (queryIter != queryNodes1.end()) {
        if ((i - tid) % nthreads == 0) {
          queryNodes.push_back(*queryIter);
          targetNodes.push_back(*targetIter);
        }
        ++i;
        ++queryIter;
        ++targetIter;
      }

      /* traverse tree */
      while (!queryNodes.empty()) {
        BOOST_AUTO(queryNode, queryNodes.back());
        BOOST_AUTO(targetNode, targetNodes.back());
        queryNodes.pop_back();
        targetNodes.pop_back();

        if (queryNode->isInternal() || targetNode->isInternal()) {
          /* should we recurse? */
          targetNode->difference(*queryNode, x);
          if (K(x) > 0.0) {
            if (queryNode->isInternal()) {
              if (targetNode->isInternal()) {
                /* split both query and target nodes */
                queryNodes.push_back(queryNode->getLeft());
                targetNodes.push_back(targetNode->getLeft());

                queryNodes.push_back(queryNode->getLeft());
                targetNodes.push_back(targetNode->getRight());

                queryNodes.push_back(queryNode->getRight());
                targetNodes.push_back(targetNode->getLeft());

                queryNodes.push_back(queryNode->getRight());
                targetNodes.push_back(targetNode->getRight());
              } else {
                /* split query node only */
                queryNodes.push_back(queryNode->getLeft());
                targetNodes.push_back(targetNode);

                queryNodes.push_back(queryNode->getRight());
                targetNodes.push_back(targetNode);
              }
            } else {
              /* split target node only */
              queryNodes.push_back(queryNode);
              targetNodes.push_back(targetNode->getLeft());

              queryNodes.push_back(queryNode);
              targetNodes.push_back(targetNode->getRight());
            }
          }
        } else {
          ++evals;
          //if (evals % nthreads == tid) {
            if (queryNode->isLeaf() && targetNode->isLeaf()) {
              i = queryNode->getIndex();
              j = targetNode->getIndex();
              x = column(transQueryX, i);
              axpy(-1.0, column(transTargetX, j), x);
              q = CUDA_EXP(lw(j) + K.logDensity(x));
              (*P)(i,tid) += q;
            } else if (queryNode->isLeaf() && targetNode->isPrune()) {
              i = queryNode->getIndex();
              const std::vector<int>& js = targetNode->getIndices();
              q = 0.0;
              for (j = 0; j < (int)js.size(); ++j) {
                x = column(transQueryX, i);
                axpy(-1.0, column(transTargetX, js[j]), x);
                q += CUDA_EXP(lw(js[j]) + K.logDensity(x));
              }
              (*P)(i,tid) += q;
            } else if (queryNode->isPrune() && targetNode->isLeaf()) {
              const std::vector<int>& is = queryNode->getIndices();
              j = targetNode->getIndex();
              for (i = 0; i < (int)is.size(); ++i) {
                x = column(transQueryX, is[i]);
                axpy(-1.0, column(transTargetX, j), x);
                q = CUDA_EXP(lw(j) + K.logDensity(x));
                (*P)(is[i],tid) += q;
              }
            } else if (queryNode->isPrune() && targetNode->isPrune()) {
              const std::vector<int>& is = queryNode->getIndices();
              const std::vector<int>& js = targetNode->getIndices();
              for (i = 0; i < (int)is.size(); ++i) {
                q = 0.0;
                for (j = 0; j < (int)js.size(); ++j) {
                  x = column(transQueryX, is[i]);
                  axpy(-1.0, column(transTargetX, js[j]), x);
                  q += CUDA_EXP(lw(js[j]) + K.logDensity(x));
                }
                (*P)(is[i],tid) += q;
              }
            //}
          }
        }
      }

      omp_set_lock(&lock);
      delete x1;
      omp_unset_lock(&lock);
    }
    sum_columns(*P, p);

    synchronize();
    delete transQueryX1;
    delete transTargetX1;
    delete P;

    omp_destroy_lock(&lock);
  }
}

template<class M1, class V1, class K1, class V2>
void bi::selfTreeDensity(KDTree<V1>& tree, const M1 X, const V1 lw,
    const K1& K, V2 p) {
  /* pre-condition */
  assert (lw.size() == X.size1());
  
  typedef typename KDTree<V1>::node_type node_type;

  BOOST_AUTO(root, tree.getRoot());
  p.clear();
  if (root != NULL) {
    std::stack<node_type> queryNodes;
    std::stack<node_type> targetNodes;
    std::stack<bool> doCrosses; // for query equals target tree optimisations

    locatable_temp_vector<ON_HOST,real>::type x(X.size2());
    int i, j;
    bool doCross;
    double d;

    queryNodes.push(root);
    targetNodes.push(root);
    doCrosses.push(false);
    
    while (!queryNodes.empty()) {
      BOOST_AUTO(queryNode, queryNodes.top());
      BOOST_AUTO(targetNode, targetNodes.top());
      queryNodes.pop();
      targetNodes.pop();
      doCross = doCrosses.top();
      doCrosses.pop();

      if (queryNode->isLeaf() && targetNode->isLeaf()) {
        i = queryNode->getIndex();
        j = targetNode->getIndex();
        x = row(X, i);
        axpy(-1.0, row(X, j), x);
        d = K.logDensity(x);
        p(i) += CUDA_EXP(lw(j) + d);
        if (doCross) {
          p(j) += CUDA_EXP(lw(i) + d);
        }
      } else if (queryNode->isLeaf() && targetNode->isPrune()) {
        i = queryNode->getIndex();
        const std::vector<int>& js = targetNode->getIndices();
        for (j = 0; j < js.size(); j++) {
          x = row(X, i);
          axpy(-1.0, row(X, js[j]), x);
          d = K.logDensity(x);
          p(i) += CUDA_EXP(lw(js[j]) + d);
          if (doCross) {
            p(js[j]) += CUDA_EXP(lw(i) + d);
          }
        }
      } else if (queryNode->isPrune() && targetNode->isLeaf()) {
        const std::vector<int>& is = queryNode->getIndices();
        j = targetNode->getIndex();
        for (i = 0; i < is.size(); i++) {
          x = row(X, is[i]);
          axpy(-1.0, row(X, j), x);
          d = K.logDensity(x);
          p(is[i]) += CUDA_EXP(lw(j) + d);
          if (doCross) {
            p(j) += CUDA_EXP(lw(is[i]) + d);
          }
        }
      } else if (queryNode->isPrune() && targetNode->isPrune()) {
        const std::vector<int>& is = queryNode->getIndices();
        const std::vector<int>& js = targetNode->getIndices();
        for (i = 0; i < is.size(); i++) {
          for (j = 0; j < js.size(); j++) {
            x = row(X, is[i]);
            axpy(-1.0, row(X, js[j]), x);
            d = K.logDensity(x);
            p(is[i]) += CUDA_EXP(lw(js[j]) + d);
            if (doCross) {
              p(js[j]) += CUDA_EXP(lw(is[i]) + d);
            }
          }
        }
      } else {
        /* should we recurse? */
        targetNode->difference(queryNode, x);
        if (K(x) > 0.0) {
          if (queryNode->isInternal()) {
            if (targetNode->isInternal()) {
              /* split both query and target nodes */
              queryNodes.push(queryNode->getLeft());
              targetNodes.push(targetNode->getLeft());
              doCrosses.push(doCross);
          
              queryNodes.push(queryNode->getLeft());
              targetNodes.push(targetNode->getRight());
              if (queryNode == targetNode) {
                /* symmetric, so just double left-right evaluation */
                doCrosses.push(true);
              } else {
                /* asymmetric, so evaluate right-left separately */
                doCrosses.push(doCross);

                queryNodes.push(queryNode->getRight());
                targetNodes.push(targetNode->getLeft());
                doCrosses.push(doCross);
              }
              
              queryNodes.push(queryNode->getRight());
              targetNodes.push(targetNode->getRight());
              doCrosses.push(doCross);
            } else {
              /* split query node only */
              queryNodes.push(queryNode->getLeft());
              targetNodes.push(targetNode);
              doCrosses.push(doCross);
          
              queryNodes.push(queryNode->getRight());
              targetNodes.push(targetNode);
              doCrosses.push(doCross);
            }
          } else {
            /* split target node only */
            queryNodes.push(queryNode);
            targetNodes.push(targetNode->getLeft());
            doCrosses.push(doCross);
        
            queryNodes.push(queryNode);
            targetNodes.push(targetNode->getRight());
            doCrosses.push(doCross);
          }
        }
      }
    }
  }
}

template<class M1, class V1, class K1, class V2>
void bi::crossTreeDensities(KDTree<V1>& tree1, KDTree<V1>& tree2,
    const M1 X1, const M1 X2, const V1 lw1, const V1 lw2, const K1& K, V2 p1,
    V2 p2,  const bool clear) {
  /* pre-condition */
  assert (lw1.size() == X1.size1());
  assert (lw2.size() == X2.size1());
  assert (p1.size() == X1.size1());
  assert (p2.size() == X2.size1());
  assert (X1.size2() == X2.size2());
  
  typedef typename KDTree<V1>::node_type node_type;

  BOOST_AUTO(root1, tree1.getRoot());
  BOOST_AUTO(root2, tree2.getRoot());
  if (clear) {
    p1.clear();
    p2.clear();
  }
  if (root1 != NULL && root2 != NULL) {
    std::stack<node_type> nodes1;
    std::stack<node_type> nodes2;

    locatable_temp_vector<ON_HOST,real>::type x(X1.size2());
    int i, j;
    double d;

    nodes1.push(root1);
    nodes2.push(root2);
    
    while (!nodes1.empty()) {
      BOOST_AUTO(node1, nodes1.top());
      BOOST_AUTO(node2, nodes2.top());
      nodes1.pop();
      nodes2.pop();

      if (node1->isLeaf() && node2->isLeaf()) {
        i = node1->getIndex();
        j = node2->getIndex();
        x = row(X1, i);
        axpy(-1.0, row(X2,j), x);
        d = K.logDensity(x);
        p1(i) += CUDA_EXP(lw2(j) + d);
        p2(j) += CUDA_EXP(lw1(i) + d);
      } else if (node1->isLeaf() && node2->isPrune()) {
        i = node1->getIndex();
        const std::vector<int>& js = node2->getIndices();
        for (j = 0; j < js.size(); ++j) {
          x = row(X1, i);
          axpy(-1.0, row(X2, js[j]), x);
          d = K.logDensity(x);
          p1(i) += CUDA_EXP(lw2(js[j]) + d);
          p2(js[j]) += CUDA_EXP(lw1(i) + d);
        }
      } else if (node1->isPrune() && node2->isLeaf()) {
        const std::vector<int>& is = node1->getIndices();
        j = node2->getIndex();
        for (i = 0; i < is.size(); ++i) {
          x = row(X1, is[i]);
          axpy(-1.0, row(X2, j), x);
          d = K.logDensity(x);
          p1(is[i]) += CUDA_EXP(lw2(j) + d);
          p2(j) += CUDA_EXP(lw1(is[i]) + d);
        }
      } else if (node1->isPrune() && node2->isPrune()) {
        const std::vector<int>& is = node1->getIndices();
        const std::vector<int>& js = node2->getIndices();
        for (i = 0; i < is.size(); ++i) {
          for (j = 0; j < js.size(); ++j) {
            x = row(X1, is[i]);
            axpy(-1.0, row(X2, js[j]), x);
            d = K(x);
            p1(is[i]) += CUDA_EXP(lw2(js[j]) + d);
            p2(js[j]) += CUDA_EXP(lw1(is[i]) + d);
          }
        }
      } else {
        /* should we recurse? */
        node2->difference(node1, x);
        if (K(x) > 0.0) {
          if (node1->isInternal()) {
            if (node2->isInternal()) {
              /* split both query and target nodes */
              nodes1.push(node1->getLeft());
              nodes2.push(node2->getLeft());
          
              nodes1.push(node1->getLeft());
              nodes2.push(node2->getRight());

              nodes1.push(node1->getRight());
              nodes2.push(node2->getLeft());
              
              nodes1.push(node1->getRight());
              nodes2.push(node2->getRight());
            } else {
              /* split query node only */
              nodes1.push(node1->getLeft());
              nodes2.push(node2);
          
              nodes1.push(node1->getRight());
              nodes2.push(node2);
            }
          } else {
            /* split target node only */
            nodes1.push(node1);
            nodes2.push(node2->getLeft());
        
            nodes1.push(node1);
            nodes2.push(node2->getRight());
          }
        }
      }
    }
  }
}

#endif
