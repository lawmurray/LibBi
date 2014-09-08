=head1 NAME

filter - filtering tasks.

=head1 SYNOPSIS

    libbi filter ...
    
=head1 INHERITS

L<Bi::Client>

=cut

package Bi::Client::filter;

use parent 'Bi::Client';
use warnings;
use strict;

=head1 OPTIONS

The C<filter> command permits the following options:

=over 4

=item C<--start-time> (default 0.0)

Start time.

=item C<--end-time> (default 0.0)

End time.

=item C<--noutputs> (default 0)

Number of dense output times. The state is always output at time
C<--end-time> and at all observation times in C<--obs-file>. This argument
gives the number of additional, equispaced times at which to output. With
C<--end-time T> and C<--noutputs K>, then for each C<k> in C<0,...,K-1>,
the state will be output at time C<T*k/K>.

=item C<--with-output-at-obs> (default 1)

Output at observation times in addition to dense output times.

=item C<--filter> (default C<bootstrap>)

The type of filter to use; one of:

=over 8

=item C<bootstrap>

Bootstrap particle filter,

=item C<lookahead>

Auxiliary particle filter with lookahead. The lookahead operates by advancing
each particle according to the L<lookahead_transition> top-level block, and
weighting according to the L<lookahead_observation> top-level block. If the
former is not defined, the L<transition> top-level block will be used
instead. If the latter is not defined, the L<observation> top-level block will
be used instead.

=item C<bridge>

Particle filter with intermediate bridging weights, as described in
Del Moral & Murray (2014). Bridging weights are assigned according to the
L<bridge> top-level block.

=begin comment

=item C<adaptive>

Bootstrap particle filter with adaptive number of particles at each time
step.

=end comment

=item C<kalman>

Extended Kalman filter. Jacobian terms are determined by symbolic
manipulations. There are some limitations to these manipulations at present,
so that some models cannot be handled. An error message will be given
in these cases.

Setting C<--filter kalman> automatically enables the 
C<--with-transform-extended> option.

=back

=back

=head2 Particle filter-specific options

The following additional options are available when C<--filter> gives a
particle filter type:

=over 4

=item C<--nparticles> (default 1)

Number of particles to use.

=item C<--ess-rel> (default 0.5)

Threshold for effective sample size (ESS) resampling trigger. Particles will
only be resampled if ESS is below this proportion of C<--nparticles>. To
always resample, use C<--ess-rel 1>. To never resample, use C<--ess-rel 0>.

=item C<--resampler> (default C<systematic>)

The type of resampler to use; one of:

=over 8

=item C<stratified>

for a stratified resampler (Kitagawa 1996),

=item C<systematic>

for a systematic (or 'deterministic stratified') resampler (Kitagawa 1996),

=item C<multinomial>

for a multinomial resampler,

=item C<metropolis>

for a Metropolis resampler (Murray 2011),

=item C<rejection>

for a rejection resampler (Murray, Lee & Jacob 2013), or

=back

=back

=head2 Stratified and multinomial resampler-specific options

=over 4

=item C<--with-sort> (default off)

Sort weights prior to resampling.

=item C<--with-kde> (default off)

Resample from a kernel density estimate of the filter density (in the style
of Liu & West 2001).

=back

=head2 Kernel density estimate options

The following additional options are available when C<--with-kde> is set.

=over 4

=item C<--b-abs> or C<--b-rel> (default 0)

Bandwidth. If C<--b-rel> is used, particles are standardised to zero mean and
unit covariance for the construction of the kernel density estimate. If
C<--b-abs> is used they are not. A value of zero in either case will result
in a rule-of-thumb bandwidth.

=item C<--with-shrink> (default on)

True to shrink the kernel density estimate to preserve covariance
(Liu & West 2001).

=back

=head2 Metropolis resampler-specific options

The following additional options are available when C<--resampler> is set to
C<metropolis>.

=over 4

=item C<-C> (default 0)

Number of steps to take.

=back

=head2 Bridge particle filter-specific options

The following additional options are available when C<--filter> is set to
C<bridge>:

=over 4

=item C<--nbridges> (default 0)

Number of dense bridge times. This argument gives the number of equispaced
times at which to assign bridge weights, and potentially resample. With
C<--end-time T> and C<--nbridges K>, then for each C<k> in C<0,...,K-1>,
brdige weighting will occur at time C<T*k/K>.

=item C<--bridge-ess-rel> (default 0.5)

Threshold for effective sample size (ESS) resampling trigger after
intermediate bridge weighting steps. See C<--ess-rel> for further
details. When sampling bridges between fully-observed states,
C<--ess-rel> should be set to 1 and C<--bridge-ess-rel> tuned
instead to minimise variance in marginal likelihood estimates.

=back

=begin comment
=head2 Adaptive particle filter-specific options

The following additional options are available when C<--filter> is set to
C<'adaptive'>:

=over 4

=item C<--stopper> (default 'deterministic')

The stopping criterion to use; one of:

=over 8

=item C<deterministic>

for a fixed number of particles,

=item C<sumofweights>

for a sum of weights,

=item C<miness>

for a minimum effective sample size (ESS),

=item C<stddev>

for a minimum standard deviation,

=item C<var>

for a minimum variance.

=back

=item C<--stopper-threshold>

Threshold value for stopping criterion.

=item C<--max-particles>

Maximum number of particles at any time, regardless of the stopping
criterion.

=item C<--block-particles>

Number of particles.

=back

=end comment

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'start-time',
      type => 'float',
      default => 0.0
    },
    {
      name => 'end-time',
      type => 'float',
      default => 0.0
    },
    {
      name => 'noutputs',
      type => 'int',
      default => 0
    },
    {
      name => 'with-output-at-obs',
      type => 'bool',
      default => 0
    },
    {
      name => 'filter',
      type => 'string',
      default => 'bootstrap'
    },
    {
      name => 'nparticles',
      type => 'int',
      default => 1
    },
    {
      name => 'ess-rel',
      type => 'float',
      default => 0.5
    },
    {
      name => 'resampler',
      type => 'string',
      default => 'systematic'
    },
    {
      name => 'with-sort',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-kde',
      type => 'bool',
      default => 0
    },
    {
      name => 'b-abs',
      type => 'float',
      default => 0.0
    },
    {
      name => 'b-rel',
      type => 'float',
      default => 1.0
    },
    {
      name => 'with-shrink',
      type => 'bool',
      default => 1
    },
    {
      name => 'C',
      type => 'int',
      default => 0
    },
    {
      name => 'nbridges',
      type => 'int',
      default => 0
    },
    {
      name => 'bridge-ess-rel',
      type => 'float',
      default => 0.5
    },
    {
      name => 'stopper',
      type => 'string',
      default => 'deterministic'
    },
    {
      name => 'stopper-threshold',
      type => 'int',
      default => 128
    },
    {
      name => 'block-particles',
      type => 'int',
      default => 128
    },
    {
      name => 'max-particles',
      type => 'int',
      default => 32768
    },
    
    # deprecations
    {
      name => 'P',
      type => 'int',
      deprecated => 1,
      message => 'use --nparticles or --nsamples instead'
    },
    {
      name => 'T',
      type => 'float',
      deprecated => 1,
      message => 'use --end-time instead'
    },
    {
      name => 'K',
      type => 'int',
      deprecated => 1,
      message => 'use --noutputs instead'
    }
);

sub init {
    my $self = shift;

    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub process_args {
    my $self = shift;

    $self->Bi::Client::process_args(@_);
    my $filter = $self->get_named_arg('filter');
    if ($filter eq 'kalman') {
        $self->set_named_arg('with-transform-extended', 1);
    }
    $self->{_binary} = 'filter';
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
