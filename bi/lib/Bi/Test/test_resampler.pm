=head1 NAME

test_resampler - test resamplers.

=head1 SYNOPSIS

    bi test_resampler ...

=head1 INHERITS

L<Bi::Client>

=cut

package Bi::Test::test_resampler;

use base 'Bi::Client';
use warnings;
use strict;

=head1 OPTIONS

=over 4

=item * C<--resampler> (default C<'stratified'>)

The type of resampler to use; one of:

=over 8

=item * C<'stratified'>

for a stratified (systematic) resampler (Kitagawa 1996),

=item * C<'multinomial'>

for a multinomial resampler,

=item * C<'metropolis'>

for a Metropolis resampler (Murray 2011),

=item * C<'rejection'>

for a rejection resampler, or

=back

=item * C<--Zs> (default 5)

Number of weight vector parameterisations to use.

=item * C<--Ps> (default 5)

Number of weight vector sizes to use.

=item * C<--reps> (default 100)

Number of trials on each combination of parameters and sizes.

=item * C<--enable-sort> (default on)

Sort weights prior to resampling.

=item * C<-C> (default 0)

Number of steps to take for Metropolis resampler.

=item * C<--output-file> (mandatory)

File to which to write output.

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
      name => 'enable-sort',
      type => 'bool',
      default => 1
    },
    {
      name => 'C',
      type => 'int',
      default => 0
    },
    {
      name => 'output-file',
      type => 'string'
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
