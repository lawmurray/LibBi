LibBi VERSION
=============

v1.0.2
------

* Removed dependency on NetCDF C++ interface, the C interface is now used
  directly.
* Added 'extended' boundary condition for dimensions.
* Added --enable-openmp/--disable-openmp command-line option.
* Added --enable-gpu-cache/--disable-gpu-cache command-line options for better
  control of GPU memory usage.
* Added --adapter-ess-rel  command-line option to avoid adaptation of proposal
  in SMC^2 when ESS too low.
* Several bug and compatibility fixes.

v1.0.1
------

* Added additional material to manual, including new section with guidance on
  tuning the proposal distribution and number of particles when using PMMH.
* Fixed sampling of joint distribution (`--target joint` now implies
  `--with-param-to-state`, just as `--target prior` and `--target prediction`
  do).
* Fixed reordering of actions and blocks when the same variable appears on
  the left more than once.
* Fixed bug in GPU implementation of multinomial resampler.
* Added `--dry-parse` option to remove parsing overhead when binaries have
  already been compiled.

v1.0.0
------

* First public release.
