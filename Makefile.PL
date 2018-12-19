use inc::Module::Install;

# metadata
name 'LibBi';
abstract 'LibBi is used for state-space modelling and Bayesian inference on modern computer hardware, including multi-core CPUs, many-core GPUs (graphics processing units) and distributed-memory clusters.';
author 'Lawrence Murray <lawrence.murray@csiro.au>';
version '1.4.4';
license 'gpl';

# dependencies
requires 'Template' => 2.24;
requires 'Graph' => 0.94;
requires 'Math::Symbolic' => 0.606;
requires 'Carp::Assert' => 0.20;
requires 'Parse::Yapp' => 0.0;
requires 'Parse::Lex' => 2.21;
requires 'File::Slurp' => 0.0;
requires 'File::ShareDir' => 1.03;
requires 'Getopt::ArgvFile' => 1.11;

# installs
install_script 'script/libbi';
install_script 'script/bi';
install_share 'share';

# ignores
makemaker_args(PL_FILES => { 'MakeParser.PL' => 'lib/Parse/Bi.pm' });
no_index 'docs';

WriteAll;
