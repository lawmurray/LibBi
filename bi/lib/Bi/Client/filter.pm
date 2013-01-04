=head1 NAME

filter - frontend to filtering client programs.

=head1 SYNOPSIS

    bi filter ...
    
=head1 INHERITS

L<Bi::Client::simulate>

=cut

package Bi::Client::filter;

use base 'Bi::Client::simulate';
use warnings;
use strict;

=head1 OPTIONS

The C<filter> program inherits all options from C<simulate>, and permits the
following additional options:

=over 4

=item * C<--filter> (default C<'pf'>)

The type of filter to use; one of:

=over 8

=item * C<'pf'> or C<'pf0'>

Bootstrap particle filter,

=item * C<'pf1'>

Auxiliary particle filter with lookahead. The lookahead operates by advancing
each particle according to the L<lookahead_transition> top-level block, and
weighting according to the L<lookahead_observation> top-level block. If the
former is not defined, the L<transition> top-level block will be used
instead. If the latter is not defined, the L<observation> top-level block will
be used instead.

=item * C<'anpf'>

Bootstrap particle filter with adaptive number of particles at each time
step.

=item * C<'ekf'>

Extended Kalman filter.

=back

=back

=head2 Particle filter-specific options

The following additional options are available when C<--filter> gives a
particle filter type:

=over 4

=item * C<--resampler> (default C<'systematic'>)

The type of resampler to use; one of:

=over 8

=item * C<'stratified'>

for a stratified resampler (Kitagawa 1996),

=item * C<'systematic'>

for a systematic (or 'deterministic stratified') resampler (Kitagawa 1996),

=item * C<'multinomial'>

for a multinomial resampler,

=item * C<'metropolis'>

for a Metropolis resampler (Murray 2011),

=item * C<'rejection'>

for a rejection resampler, or

=item * C<'kernel'>

for a kernel density resampler (Liu & West 2001).

=back

=item * C<--ess-rel> (default 1)

Use effective sample size (ESS) resampling condition. Particles will only be
resampled if ESS is below this proportion of C<P>.

=back

=head2 Adaptive particle filter-specific options

The following additional options are available when C<--filter> is set to
C<'anpf'>:

=over 4

=item * C<--stopper> (default 'miness')

The stopping criterion to use; one of:

=over 8

=item * C<'deterministic'>

fixed number of particles,

=item * C<'sumofweights'>

sum of weights,

=item * C<'miness'>

minimum ESS,

=item * C<'stddev'>

...

=item * C<'var'>

...

=back

=item C<--rel-threshold>

...

=item C<--block-P>

...

=item C<--min-ess-rel>

...

=item C<--max-P>

...

=back

=head2 Stratified and multinomial resampler-specific options

=over 4

=item * C<--enable-sort> (default on)

Sort weights prior to resampling.

=back

=head2 Kernel resampler-specific options

=over 4

=item * C<--b-abs> or C<--b-rel> (default 0)

Bandwidth. If C<--b-rel> is used, particles are standardised to zero mean and
unit covariance for the construction of the kernel density estimate. If
C<--b-abs> is used they are not. A value of zero in either case will result
in a rule-of-thumb bandwidth.

=item * C<--enable-shrink> (default on)

True to shrink the kernel density estimate to preserve covariance
(Liu & West 2001).

=back

=head2 Metropolis resampler-specific options

=over 4

=item * C<-C> (default 0)

Number of steps to take.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'filter',
      type => 'string',
      default => 'pf'
    },
    {
      name => 'resampler',
      type => 'string',
      default => 'systematic'
    },
    {
      name => 'ess-rel',
      type => 'float',
      default => 1.0
    },
    {
      name => 'enable-sort',
      type => 'bool',
      default => 1
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
      name => 'enable-shrink',
      type => 'bool',
      default => 1
    },
    {
      name => 'C',
      type => 'int',
      default => 0
    },
    {
      name => 'stopper',
      type => 'string',
      default => 'miness'
    },
    {
      name => 'block-P',
      type => 'int',
      default => 128
    },
    {
      name => 'rel-threshold',
      type => 'int',
      default => 10
    },
    {
      name => 'max-P',
      type => 'int',
      default => 32768
    }
);

sub init {
    my $self = shift;

    Bi::Client::simulate::init($self);
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub process_args {
    my $self = shift;

    $self->Bi::Client::simulate::process_args(@_);
    my $filter = $self->get_named_arg('filter');
    my $binary;
    if ($filter eq 'ekf') {
        $self->set_named_arg('transform-extended', 1);
        $binary = 'ekf';
    } else {
        $binary = 'pf';
    }
    $self->{_binary} = $binary;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
