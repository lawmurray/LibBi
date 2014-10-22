/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * $Rev $
 * $Date$
 */
#include "SMCNullBuffer.hpp"

bi::SMCNullBuffer::SMCNullBuffer(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    MCMCNullBuffer(m, P, T, file, mode, schema) {
  //
}
