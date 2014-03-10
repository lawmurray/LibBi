=head1 NAME

eval_ - arbitrary expression.

=head1 SYNOPSIS

    x <- some_expression

=head1 DESCRIPTION

An C<eval_> action sets a variable using an expression. It need not be used
explicitly: any expression using the C<<-> operator without naming an action
is evaluated using C<eval_>.

=cut

package Bi::Action::eval_;

use parent 'Bi::Action';
use warnings;
use strict;

use Carp::Assert;

=head1 PARAMETERS

=over 4

=item C<expr> (position 0, mandatory)

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
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');
    
    $self->set_parent('eval_');
    $self->set_can_combine(1);
    $self->set_can_nest(1);
    $self->set_unroll_args(0);
    $self->set_shape($self->get_named_arg('expr')->get_shape);
    unless ($self->get_shape->get_count == 0 || $self->get_left->get_shape->equals($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }
}

sub mean {
    my $self = shift;
    return $self->get_named_arg('expr')->clone;
}

sub std {
    #
}

sub jacobian {
    my $self = shift;
    
    my $expr = $self->get_named_arg('expr');
    my @refs = @{$expr->get_all_var_refs};
    my @Js = map { $expr->d($_) } @refs;

    return (\@Js, \@refs);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
