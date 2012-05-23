=head1 NAME

eval - generic update action.

=head1 SYNOPSIS

    x <- some_expression
    x <- eval(some_expression)
    x <- eval(expr = some_expression)

=head1 DESCRIPTION

An C<eval> action sets a variable using an expression. It need not be used
explicitly: any expression not explicitly enclosed by a named action is
evaluated using C<eval>.

=cut

package Bi::Action::eval;

use base 'Bi::Model::Action';
use warnings;
use strict;

use Carp::Assert;

=head1 PARAMETERS

=over 4

=item * C<expr> (position 0, mandatory)

Expression to evaluate.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'expr',
    positional => 1,
    mandatory => 1
  }
];

sub validate {
    my $self = shift;
    
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');
    $self->set_parent('eval');
    $self->set_can_combine(1);
    $self->set_can_nest(1);
    $self->set_unroll_args(0);
    $self->set_dims($self->get_named_arg('expr')->get_dims);
}

sub mean {
    my $self = shift;
    return $self->get_named_arg('expr')->clone;
}

sub jacobian {
    my $self = shift;
    
    my $expr = $self->get_named_arg('expr');
    my @refs = (@{$expr->get_vars('noise')}, @{$expr->get_vars('state')});
    my @J = map { $expr->d($_) } @refs;

    return (\@J, \@refs);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
