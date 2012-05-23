/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_LOCATION_HPP
#define BI_MISC_LOCATION_HPP

namespace bi {
/**
 * Tags for location-specific data types and operations.
 */
enum Location {
  /**
   * Locate object on host.
   */
  ON_HOST = 0,

  /**
   * Locate object on device.
   */
  ON_DEVICE = 1
};
}

#endif
