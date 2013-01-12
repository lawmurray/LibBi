use inc::Module::Install;

# metadata
name 'LibBi';
abstract 'Bayesian inference for state-space models, including sequential
Monte Carlo (SMC), particle Markov chain Monte Carlo (PMCMC) and SMC^2 methods
on multithreaded, graphics processing unit (GPU) and distributed memory
architectures.';
author 'Lawrence Murray <lawrence.murray@csiro.au>';
version '1.0.0';
license 'gpl';

# dependencies
requires 'Template' => 2.24;
requires 'Math::Symbolic' => 0.606;
requires 'Carp::Assert' => 0.20;
requires 'Parse::Yapp' => 0.0;
requires 'Parse::Lex' => 2.21;
requires 'File::Slurp' => 0.0;
requires 'File::ShareDir' => 1.03;
requires 'Getopt::ArgvFile' => 1.11;

# installs
install_script 'script/bi';
install_share 'share';

# ignores
makemaker_args(PL_FILES => { 'MakeParser.PL' => '/lib/Parse/Bi.pm' });
no_index 'docs';

WriteAll;
