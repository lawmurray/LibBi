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

=item C<--nsamples> (default 1)

Number of parameter samples to draw.

=item C<--sampler> (default C<'pmmh'>)

The type of sampler to use; one of:

=over 8

=item C<'pmmh'>

Particle marginal Metropolis-Hastings (PMMH). The proposal works according to
the L<proposal_parameter> top-level block. If this is not defined,
independent draws are taken from the L<proposal> top-level block instead. If
C<--transform-initial-to-param> is on, the L<proposal_initial> top-level
block is used to make Metropolis-Hastings proposals over initial conditions
also. If this is not defined, independent draws are taken from the L<initial>
top-level block instead.

=item C<'smc2'>

Sequential Monte Carlo Squared (SMC^2).

=back

=item C<--conditional-pf>

...

=item C<--joint-adaptation>

...

=back

=head2 SMC2-specific options

=over 4

=item C<--nmoves> (default 1)

Number of PMMH steps to perform after resampling.

=item C<--sample-ess-rel> (default 0.5)

ESS threshold triggering resampling steps. Parameter samples will only be
resampled if ESS is below this proportion of C<--nsamples>.

=item C<--adapter> (default 'none')

Adaptation strategy for rejuvenation proposals:

=over 8

=item C<'none'>

No adaptation.

=item C<'local'>

Local proposal adaptation.

=item C<'global'>

Global proposal adaptation.

=back

=item C<--adapter-scale> (default 0.25)

When local proposal adaptation is used, the scaling factor of the local
proposal standard deviation relative to the global sample standard deviation.

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
      name => 'adapter',
      type => 'string',
      default => 'none'
    },
    {
      name => 'adapter-scale',
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

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
