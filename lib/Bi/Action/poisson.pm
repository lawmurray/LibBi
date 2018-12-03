=head1 NAME

poisson - Poisson distribution.

=head1 SYNOPSIS

    x ~ poisson()
    x ~ poisson(1.0)
    x ~ poisson(rate = 2.0)

=head1 DESCRIPTION

A C<poisson> action specifies that a variable is Poisson distributed according to
the given C<rate> parameter. Note that the implementation will evaluate
densities for any (not necessarily integer) x. It is left to the user to ensure
consistency (e.g., using this only with integer observations).

=cut

package Bi::Action::poisson;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<rate> (position 0, default 1.0)

Rate parameter of the distribution.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'rate',
    positional => 1,
    default => 1.0
  }
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('rate');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;

    my $mean = $self->get_named_arg('rate');

    return $mean;
}

1;

=head1 AUTHOR

Sebastian Funk <sebastian.funk@lshtm.ac.uk>

