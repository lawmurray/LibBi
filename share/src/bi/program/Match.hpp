/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_MATCH_HPP
#define BI_PROGRAM_MATCH_HPP

#include <vector>

#include "boost/shared_ptr.hpp"

namespace biprog {
class Expression;

/**
 * Match between expressions.
 *
 * @ingroup program
 */
class Match {
public:
  /**
   * Scores.
   */
  enum Score {
    /**
     * Score for match at leaf/concrete level.
     */
    SCORE_LEAF,

    /**
     * Score for match at declaration level.
     */
    SCORE_DECLARATION,

    /**
     * Score for match at statement level.
     */
    SCORE_STATEMENT,

    /**
     * Score for match at expression level.
     */
    SCORE_EXPRESSION
  };

  /**
   * Comparison operator for sorting matches by score.
   */
  bool operator<(const Match& o) const;

  /**
   * Add a match.
   *
   * @param arg The argument.
   * @param param The formal parameter.
   * @param score The score.
   */
  void push(boost::shared_ptr<Expression> arg,
      boost::shared_ptr<Expression> param, const Score score);

  /**
   * Clear for reuse.
   */
  void clear();

private:
  /**
   * Arguments.
   */
  std::vector<boost::shared_ptr<Expression> > args;

  /**
   * Parameters.
   */
  std::vector<boost::shared_ptr<Expression> > params;

  /**
   * Scores.
   */
  std::vector<int> scores;
};
}

#include "../misc/assert.hpp"

#include "boost/typeof/typeof.hpp"

inline bool biprog::Match::operator<(const Match& o) const {
  /* pre-condition */
  BI_ASSERT(scores.size() == o.scores.size());

  BOOST_AUTO(iter1, scores.begin());
  BOOST_AUTO(iter2, o.scores.begin());
  bool result = scores.size() > 0;
  while (result && iter1 != scores.end()) {
    result = *iter1 < *iter2;
    ++iter1;
    ++iter2;
  }
  return result;
}

inline void biprog::Match::push(boost::shared_ptr<Expression> arg,
    boost::shared_ptr<Expression> param, const Score score) {
  args.push_back(arg);
  params.push_back(param);
  scores.push_back(score);
}

inline void biprog::Match::clear() {
  args.clear();
  params.clear();
  scores.clear();
}

#endif
