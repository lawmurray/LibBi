=head1 NAME

truncated_gaussian - truncated Gaussian distribution.

=head1 SYNOPSIS

    x ~ truncated_gaussian(0.0, 1.0, -2.0, 2.0)
    x ~ truncated_gaussian(0.0, 1.0, lower = -2.0, upper = 2.0)
    x ~ truncated_gaussian(0.0, 1.0, upper = 2.0)

=head1 DESCRIPTION

A C<truncated_gaussian> action specifies that a variable is distributed
according to a Gaussian distribution with a closed lower and/or upper bound.

For a one-sided truncation, simply omit the relevant C<lower> or C<upper>
argument.

The current implementation uses a naive rejection sampling with the full
Gaussian distribution used as a proposal. The rejection rate is simply the
area under the Gaussian curve between C<lower> and C<upper>. If this is
significantly less than one, the rejection rate will be high, and performance
slow.

=cut

package Bi::Action::truncated_gaussian;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<mean> (position 0, default 0.0)

Mean.

=item C<std> (position 1, default 1.0)

Standard deviation.

=item C<lower> (position 2)

Lower bound.

=item C<upper> (position 3)

Upper bound.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'mean',
    positional => 1,
    default => 0.0
  },
  {
    name => 'std',
    positional => 1,
    default => 1.0
  },
  {
    name => 'lower',
    positional => 1
  },
  {
    name => 'upper',
    positional => 1
  }
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->set_name('truncated_gaussian'); # synonyms exist, standardise name
    $self->ensure_op('~');
    $self->ensure_scalar('mean');
    $self->ensure_scalar('std');
    $self->ensure_scalar('lower');
    $self->ensure_scalar('upper');
    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
