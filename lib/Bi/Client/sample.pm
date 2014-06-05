=head1 NAME

sample - sample the prior, joint or posterior distribution.

=head1 SYNOPSIS

    libbi sample --target prior ...
    
    libbi sample --target joint ...
    
    libbi sample --target posterior ...
    
=head1 INHERITS

L<Bi::Client::filter>

=cut

package Bi::Client::sample;

use parent 'Bi::Client::filter';
use warnings;
use strict;

=head1 OPTIONS

The C<sample> command inherits all options from C<filter>, and permits the
following additional options:

=over 4

=item C<--target> (default C<posterior>)

The target distribution to sample; one of:

=over 8

=item C<prior>

To sample from the prior distribution.

=item C<joint>

To sample from the joint distribution. This is equivalent to
C<--target prior --with-transform-obs-to-state>.

=item C<posterior>

To sample from the posterior distribution. Use C<--obs-file> to provide
observations.

=item C<prediction>

To sample forward in time from a given initial state. Use C<--init-file> to
set the initial state, and this initial state determines the interpretation
of the prediction. For example, the C<--init-file> may be the output
file of a previous sampling of the posterior distribution, in which case the
result is a posterior prediction.

=back

=item C<--sampler> (default C<mh>)

The type of sampler to use for C<--target posterior>; one of:

=over 8

=item C<mh> or (deprecated) C<pmmh>

Marginal Metropolis-Hastings.

=item C<sir> or (deprecated) C<smc2>

Marginal sequential importance resampling.

=item C<srs>

Marginal sequential rejection sampling.

=back

For PMMH, the proposal works according to the L<proposal_parameter> top-level
block in the model. If this is not defined, independent draws are taken from
the L<parameter> top-level block instead. If
C<--with-transform-initial-to-param> is used, the L<proposal_initial>
top-level block is used to make Metropolis-Hastings proposals over initial
conditions also. If this is not defined, independent draws are taken from the
L<initial> top-level block instead.

For SMC^2, the same blocks are used as proposals for rejuvenation steps,
unless one of the adaptation strategies below is enabled.

=item C<--nsamples> (default 1)

Number of samples to draw.

=back

=head2 MarginalSIR-specific options

=over 4

=item C<--nmoves> (default 1)

Number of PMMH steps to perform after resampling.

=item C<--sample-ess-rel> (default 0.5)

Threshold for effective sample size (ESS) resampling trigger. Parameter
particles will only be resampled if ESS is below this proportion of
C<--nsamples>. To always resample, use C<--sample-ess-rel 1>. To never
resample, use C<--sample-ess-rel 0>.

=item C<--adapter> (default none)

Adaptation strategy for rejuvenation proposals:

=over 8

=item C<none>

No adaptation.

=item C<local>

Local proposal adaptation.

=item C<global>

Global proposal adaptation.

=back

=item C<--adapter-scale> (default 0.25)

When local proposal adaptation is used, the scaling factor of the local
proposal standard deviation relative to the global sample standard deviation.

=item C<--adapter-ess-rel> (default 0.0)

Threshold for effective sample size (ESS) adaptation trigger. Adaptation will
only be performed if ESS is above this proportion of C<--nsamples>. To always
adapt, use C<--adapter-ess-rel 1>. If adaptation is not performed, the
C<proposal_parameter> top-level block is used for rejuvenation proposals
instead. 

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'target',
      type => 'string',
      default => 'posterior'
    },
    {
      name => 'sampler',
      type => 'string',
      default => 'mh'
    },
    {
      name => 'nsamples',
      type => 'int',
      default => 1
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
    {
      name => 'adapter-ess-rel',
      type => 'float',
      default => 0.0
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

    my $target = $self->get_named_arg('target');
    my $sampler = $self->get_named_arg('sampler');
    my $filter = $self->get_named_arg('filter');
    
    if ($target eq 'prior' || $target eq 'prediction') {
        if (!$self->is_named_arg('with-transform-param-to-state')) {
	        $self->set_named_arg('with-transform-param-to-state', 1);
        }
    } elsif ($target eq 'joint') {
        if (!$self->is_named_arg('with-transform-param-to-state')) {
	        $self->set_named_arg('with-transform-param-to-state', 1);
        }
        if (!$self->is_named_arg('with-transform-obs-to-state')) {
            $self->set_named_arg('with-transform-obs-to-state', 1);
        }
    } else {
    	if ($sampler eq 'sir' || $sampler eq 'smc2') {
	    	$self->set_named_arg('sampler', 'sir'); # standardise name
    	}
    }
    
    $self->{_binary} = 'sample';
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
