=head1 NAME

negbin - Negative binomial distribution.

=head1 SYNOPSIS

    x ~ negbin()
    x ~ negbin(1.0, 2.0)
    x ~ negbin(mean = 1.0, shape = 2.0)

=head1 DESCRIPTION

A C<negbin> action specifies that a variable is distributed according to
a negative binomial distribution with the given C<mean> and C<shape> parameters.
Note that the implementation will evaluate densities for any (not necessarily
integer) x. It is left to the user to ensure consistency (e.g., using this only
with integer observations).


=cut

package Bi::Action::negbin;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<mean> (position 0, default 1.0)

Mean parameter of the distribution.

=item C<shape> (position 1, default 1.0)

Shape parameter of the distribution.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'mean',
    positional => 1,
    default => 1.0
  },
  {
    name => 'shape',
    positional => 1,
    default => 1.0
  },
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('mean');
    $self->ensure_scalar('shape');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;

    my $mean = $self->get_named_arg('mean');

    return $mean;
}

1;

=head1 AUTHOR

Sebastian Funk <sebastian.funk@lshtm.ac.uk>

=head1 VERSION

$Rev$ $Date$
