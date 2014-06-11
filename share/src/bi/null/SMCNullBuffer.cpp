/**
 * @file
 *
 * @author Pierre Jacob <jacob@ceremade.dauphine.fr>
 * $Rev $
 * $Date$
 */
#include "SMCNullBuffer.hpp"

bi::SMCNullBuffer::SMCNullBuffer(const Model& m, const std::string& file,
    const FileMode mode, const SchemaMode schema, const size_t P,
    const size_t T) :
    MCMCNullBuffer(m, file, mode, schema, P, T) {
  //
}
