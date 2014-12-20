=head1 NAME

test_resampler - test resamplers.

=head1 SYNOPSIS

    libbi test_resampler ...

=head1 INHERITS

L<Bi::Client>

=cut

package Bi::Test::test_resampler;

use parent 'Bi::Client';
use warnings;
use strict;

=head1 OPTIONS

=over 4

=item C<--resampler> (default C<'stratified'>)

The type of resampler to use; one of:

=over 8

=item C<'stratified'>

for a stratified (systematic) resampler (Kitagawa 1996),

=item C<'multinomial'>

for a multinomial resampler,

=item C<'metropolis'>

for a Metropolis resampler (Murray 2011),

=item C<'rejection'>

for a rejection resampler, or

=back

=item C<--Zs> (default 5)

Number of weight vector parameterisations to use.

=item C<--Ps> (default 5)

Number of weight vector sizes to use.

=item C<--reps> (default 100)

Number of trials on each combination of parameterisations and sizes.

=item C<--with-copy> (default off)

Copy weights to host/device and ancestors back to host/device as part of
test and timing.

=item C<--with-cuda> (default off)

Use this to actually run CUDA code, C<--enable-cuda> will not achieve this
automatically for C<test_resampler>, as the intention may be to time
resampling on CPU when copying weight vectors from GPU.

=item C<-C> (default 1)

Divisor under the default number of steps in the Metropolis resampler.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'resampler',
      type => 'string',
      default => 'stratified'
    },
    {
      name => 'Zs',
      type => 'int',
      default => 5
    },
    {
      name => 'Ps',
      type => 'int',
      default => 5
    },
    {
      name => 'reps',
      type => 'int',
      default => 100
    },
    {
      name => 'with-sort',
      type => 'bool',
      deprecated => 1
    },
    {
      name => 'with-copy',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-cuda',
      type => 'bool',
      default => 0
    },
    {
      name => 'C',
      type => 'int',
      default => 1
    }
);

sub init {
    my $self = shift;

	$self->{_binary} = 'test_resampler';
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub needs_model {
    return 0;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
