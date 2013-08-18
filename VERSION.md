LibBi VERSION
=============

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
