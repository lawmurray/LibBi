=head1 NAME

sample - frontend to sampling client programs.

=head1 SYNOPSIS

    bi sample ...
    
=head1 INHERITS

L<Bi::Client::filter>

=cut

package Bi::Client::sample;

use base 'Bi::Client::filter';
use warnings;
use strict;

=head1 OPTIONS

The C<sample> program inherits all options from C<filter>, and permits the
following additional options:

=over 4

=item * C<--nsamples> (default 1)

Number of parameter samples to draw.

=item * C<--filter-file>

File from which to read and write intermediate filter results.

=item * C<--include-initial> (default 0)

Include initial conditions in outer Metropolis-Hastings loop (as opposed to
inner filtering loop)?

=item * C<--sampler> (default C<'pmmh'>)

The type of sampler to use; one of:

=over 8

=item * C<'pmmh'>

Particle marginal Metropolis-Hastings (PMMH). The proposal works according to
the L<proposal_parameter> top-level block. If this is not defined,
independent draws are taken from the L<proposal> top-level block instead. If
C<--transform-initial-to-param> is on, the L<proposal_initial> top-level
block is used to make Metropolis-Hastings proposals over initial conditions
also. If this is not defined, independent draws are taken from the L<initial>
top-level block instead.

=item * C<'smc2'>

Sequential Monte Carlo Squared (SMC^2).

=back

=item * C<--conditional-pf>

...

=item * C<--joint-adaptation>

...

=back

=head2 SMC2-specific options

=over 4

=item * C<--nmoves> (default 1)

Number of PMMH steps to perform after resampling.

=item * C<--sample-ess-rel> (default 0.5)

ESS threshold triggering resampling steps. Parameter samples will only be
resampled if ESS is below this proportion of C<--nsamples>.

=item * C<--enable-local-move> (default off)

Enable random walk proposals, instead of independent proposals

=item * C<--local-move-scale> (default 0.25)

The proportion of the parameter samples' standard deviation to be used in the
proposal distribution for random walk proposals.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'nsamples',
      type => 'int',
      default => 1
    },
    {
      name => 'sampler',
      type => 'string',
      default => 'pmmh'
    },
    {
      name => 'filter-file',
      type => 'string'
    },
    {
      name => 'include-initial',
      type => 'int',
      default => 0
    },
    {
      name => 'conditional-pf',
      type => 'int',
      default => '0'
    },
    {
      name => 'joint-adaptation',
      type => 'int',
      default => '0'
    },
        {
      name => 'nmoves',
      type => 'int',
      default => '1'
    },
    {
      name => 'sample-ess-rel',
      type => 'float',
      default => 0.5
    },
    {
      name => 'enable-local-move',
      type => 'bool',
      default => 0
    },
    {
      name => 'local-move-scale',
      type => 'float',
      default => 0.5
    },
);

sub init {
    my $self = shift;

    Bi::Client::filter::init($self);
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub process_args {
    my $self = shift;

    $self->Bi::Client::filter::process_args(@_);
    my $sampler = $self->get_named_arg('sampler');
    my $binary;
    if ($sampler eq 'smc2') {
        $binary = 'smc2';
    } else {
        $binary = 'pmmh';
    }
    $self->{_binary} = $binary;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
