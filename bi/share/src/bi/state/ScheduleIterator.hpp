/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_SCHEDULEITERATOR_HPP
#define BI_STATE_SCHEDULEITERATOR_HPP

#include "ScheduleElement.hpp"

#include <vector>

namespace bi {
/**
 * Iterator over Schedule.
 *
 * @ingroup state
 */
typedef std::vector<ScheduleElement>::const_iterator ScheduleIterator;
}

#endif
