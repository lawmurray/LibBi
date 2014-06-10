/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_BUFFER_HPP
#define BI_BUFFER_BUFFER_HPP

namespace bi {
/**
 * Schema flags.
 */
enum SchemaMode {
  /**
   * Default schema.
   */
  DEFAULT,

  /**
   * Have multiple parameter samples.
   */
  MULTI,

  /**
   * Multiple parameter samples, but parameters only.
   */
  PARAM_ONLY,

  /**
   * Use flexi schema.
   */
  FLEXI
};

/**
 * File open flags.
 */
enum FileMode {
  /**
   * Open file read-only.
   */
  READ_ONLY,

  /**
   * Open file for reading and writing,
   */
  WRITE,

  /**
   * Open file for reading and writing, replacing any existing file of the
   * same name.
   */
  REPLACE,

  /**
   * Open file for reading and writing, fails if any existing file of the
   * same name
   */
  NEW
};
}

#endif
