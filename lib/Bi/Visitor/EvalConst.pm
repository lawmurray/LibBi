=head1 NAME

Bi::Visitor::EvalConst - visitor for evaluating a constant expression.

=head1 SYNOPSIS

    use Bi::Visitor::EvalConst;
    $val = Bi::Visitor::EvalConst->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::EvalConst;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;
use Bi::Visitor::ToPerl;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr>

L<Bi::Expression> object.

=back

Returns the result of evaluating the expression.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;

    assert ($expr->isa('Bi::Expression') && $expr->is_const) if DEBUG;

    my $perl = Bi::Visitor::ToPerl->evaluate($expr);
    
    return eval($perl);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
