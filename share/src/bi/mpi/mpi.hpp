/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_MPI_HPP
#define BI_MPI_MPI_HPP

#include <string>

namespace bi {
/**
 * Append rank to file name.
 *
 * @param name File name.
 */
std::string append_rank(const std::string& name);

}

#endif
