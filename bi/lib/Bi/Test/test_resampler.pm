=head1 NAME

test_resampler - test resamplers.

=head1 SYNOPSIS

    bi test_resampler ...

=head1 INHERITS

L<Bi::Client>

=cut

package Bi::Test::test_resampler;

use base 'Bi::Client::filter';
use warnings;
use strict;

=head1 OPTIONS

The C<test_resampler> program inherits all options from C<filter> (largely
for convenience, for resampling options), and permits the following
additional options:

=over 4

=item * C<--Zs> (default 5)

Number of weight vector parameterisations to use.

=item * C<--Ps> (default 5)

Number of weight vector sizes to use.

=item * C<--reps> (default 100)

Number of trials on each combination of parameters and sizes.

=cut
our @CLIENT_OPTIONS = (
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
    }
);

sub init {
    my $self = shift;

    Bi::Client::filter::init($self);
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub process_args {
	my $self = shift;
	
	$self->Bi::Client::filter::process_args(@_);
	$self->{_binary} = 'test_resampler';
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
