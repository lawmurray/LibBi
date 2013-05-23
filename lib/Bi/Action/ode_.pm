=head1 NAME

ode_ - ordinary differential equation (ODE) action.

=head1 SYNOPSIS

    dx/dt = expr
    x = ode_(expr)
    x = ode_(dfdt = expr)
    
=head1 DESCRIPTION

An C<ode_> specifies an ordinary differential equation for the update of
a variable via the numerical simulation of ODEs. It need not be used
explicitly, any action using the C<dx/dt = ...> differential equation syntax
automatically uses this action.

An C<ode_> action may only be used within an L<ode> block, and may only be
applied to a scalar variable.

=cut

package Bi::Action::ode_;

use parent 'Bi::Action';
use warnings;
use strict;

use Carp::Assert;

=head1 PARAMETERS

=over 4

=item C<dfdt> (position 0, mandatory)

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
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    
    # check removed for backwards compatibility with model files written
    # prior to introduction of differential equation syntax
    #$self->ensure_op('=');

    $self->set_parent('ode');
    $self->set_can_combine(1);
    $self->set_is_inplace(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;
    return new Bi::Expression::Function('ode_',
        [ $self->get_named_arg('dfdt')->clone ]);
}

sub std {
    return undef;
}

sub jacobian {
    my $self = shift;
    
    my $expr = $self->get_named_arg('dfdt');
    my @refs = @{$expr->get_all_var_refs};
    my @J = map { $expr->d($_) } @refs;
    
    return (\@J, \@refs);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
