=head1 NAME

ode - ordinary differential equation (ODE) action.

=head1 SYNOPSIS

    x <- ode(expr)
    x <- ode(dfdt = expr)
    
=head1 DESCRIPTION

An C<ode> action updates a variable via the numerical simulation of ODEs.

An ode action may only be used within an L<ode> block, and may only be
applied to a scalar variable.

=cut

package Bi::Action::ode;

use base 'Bi::Model::Action';
use warnings;
use strict;

use Carp::Assert;

=head1 PARAMETERS

=over 4

=item * C<dfdt> (position 0, mandatory)

An L<expression> giving the time derivative of the variable.

=back

=cut
our $ACTION_ARGS = [
{
    name => 'dfdt',
    positional => 1,
    mandatory => 1
  }
];

sub validate {
    my $self = shift;
    
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');

    $self->set_parent('ode');
    $self->set_can_combine(1);
    $self->set_is_inplace(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;
    return new Bi::Expression::Function('ode', [], {
        'dfdt' => $self->get_named_arg('dfdt')->clone
    });
}

sub std {
    return undef;
}

sub jacobian {
    my $self = shift;
    
    my $expr = $self->get_named_arg('dfdt');
    my @refs = @{$expr->get_vars};
    my @J = map { $expr->d($_) } @refs;
    
    return (\@J, \@refs);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
