=head1 NAME

betabin - Beta-binomial distribution.

=head1 SYNOPSIS

    x ~ betabin()
    x ~ betabin(10, 1.0, 2.0)
    x ~ betabin(n=10, alpha = 1.0, beta = 2.0)

=head1 DESCRIPTION

A C<betabin> action specifies that a variable is distributed according to
a Beta-binomial distribution with the given number of trials C<n>, and
C<alpha> and C<beta> parameters. Note that the implementation will evaluate densities for any (not necessarily integer) x. It is left to the user to ensure consistency (e.g., using this only with integer observations).


=cut

package Bi::Action::betabin;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<n> (position 0, default 1)

Number of trials

=item C<alpha> (position 1, default 1.0)

First shape parameter of the distribution.

=item C<beta> (position 2, default 1.0)

Second shape parameter of the distribution.

=back

=cut
our $ACTION_ARGS = [
    {
        name => 'n',
        positional => 1,
        default => 1
    },
    {
        name => 'alpha',
        positional => 1,
        default => 1.0
    },
    {
        name => 'beta',
        positional => 1,
        default => 1.0
    },
];

sub validate {
    my $self = shift;

    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('n');
    $self->ensure_scalar('alpha');
    $self->ensure_scalar('beta');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;

    my $n = $self->get_named_arg('n')->clone;
    my $alpha = $self->get_named_arg('alpha')->clone;
    my $beta = $self->get_named_arg('beta')->clone;
    my $mean = $n * $alpha/($alpha + $beta);

    return $mean;
}

1;

=head1 AUTHOR

Sebastian Funk <sebastian.funk@lshtm.ac.uk>

=head1 VERSION

$Rev$ $Date$
