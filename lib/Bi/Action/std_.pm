=head1 NAME

std_ - Internal action for updating square-root covariance variables with
intrinsic variability.

=head1 SYNOPSIS

    Q_x_x_ ~ std_(expr)

=head1 DESCRIPTION

A C<std_> action is used to specify the intrinsic variability of a variable
-- that part not dependent on any other variable.

=cut

package Bi::Action::std_;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<std> (position 0, mandatory)

For a univariate or i.i.d. distribution, a scalar giving the standard
deviation. For a multivariate distribution of independent variables, a vector
of the same size as the target giving the standard deviations of those
variables. For a general multivariate distribution, a matrix square-root
of the covariance matrix.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'std',
    positional => 1,
    mandatory => 1
  }
];

sub validate {
    my $self = shift;

    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');

    my $const_std = $self->get_named_arg('std')->is_const;
    my $common_std = $self->get_named_arg('std')->is_common;
    my $vector_std = $self->get_named_arg('std')->is_scalar || $self->get_named_arg('std')->is_vector;

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    if ($const_std) {
        $self->set_parent('const_std_');
    } elsif ($common_std) {
        if ($vector_std) {
            $self->set_parent('common_orthogonal_std_');
        } else {
            $self->set_parent('common_std_');
        }
    } else {
        if ($vector_std) {
            $self->set_parent('orthogonal_std_');
        } else {
            $self->set_parent('std_');
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

