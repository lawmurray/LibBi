=head1 NAME

ode - system of ordinary differential equations.

=head1 SYNOPSIS

    ode(alg = 'RK4(3)', h = 1.0, atoler = 1.0e-3, rtoler = 1.0e-3) {
      dx/dt = ...
      dy/dt = ...
      ...
    }

    ode('RK4(3)', 1.0, 1.0e-3, 1.0e-3) {
      dx/dt = ...
      dy/dt = ...
      ...
    }

=head1 DESCRIPTION

An C<ode> block is used to group multiple ordinary differential equations
into one system, and configure the numerical integrator used to simulate
them.

An C<ode> block may not contain nested blocks, and may only contain
ordinary differential equation actions.

=cut

package Bi::Block::ode;

use parent 'Bi::Block';
use warnings;
use strict;

use Carp::Assert;

use Bi::Utility qw(find);

=head1 PARAMETERS

=over 4

=item C<alg> (position 0, default C<'RK4(3)'>)

The numerical integrator to use. Valid values are:

=over 8

=item C<'RK4'>

The classic order 4 Runge-Kutta with fixed step size.

=item C<'RK5(4)'>

An order 5(4) Dormand-Prince with adaptive step size.

=item C<'RK4(3)'>

An order 4(3) low-storage Runge-Kutta with adaptive step size.

=back

=item C<h> (position 1, default 1.0)

For a fixed step size, the step size to use. For an adaptive step size, the
suggested initial step size to use.

=item C<atoler> (position 2, default 1.0e-3)

The absolute error tolerance for adaptive step size control.

=item C<rtoler> (position 3, default 1.0e-3)

The relative error tolerance for adaptive step size control.

=back

=cut
our $BLOCK_ARGS = [
  {
    name => 'alg',
    positional => 1,
    default => "RK4(3)"
  },
  {
    name => 'h',
    positional => 1,
    default => 1.0
  },
  {
    name => 'atoler',
    positional => 1,
    default => 0.001
  },
  {
    name => 'rtoler',
    positional => 1,
    default => 0.001
  }
];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    my $alg = $self->get_named_arg('alg')->eval_const;
    if ($alg ne 'RK4' && $alg ne 'RK5(4)' && $alg ne 'RK4(3)') {
        die("unrecognised value '$alg' for argument 'alg' of block 'ode'\n");
    }
    
    foreach my $action (@{$self->get_actions}) {
        if ($action->get_name ne 'ode_') {
            die("an 'ode' block may only contain ordinary differential equation actions\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
