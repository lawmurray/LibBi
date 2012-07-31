=head1 NAME

Bi::Visitor::Standardiser - visitor for standardising expressions in
a model, replacing higher-level operators with function calls, etc.

=head1 SYNOPSIS

    use Bi::Visitor::Standardiser;
    Bi::Visitor::Standardiser->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::Standardiser;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<model>)

Evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=back

No return value.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = new Bi::Visitor; 
    bless $self, $class;
    
    $model->accept($self);
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $result = $node;
    
    if ($node->isa('Bi::Expression::UnaryOperator')) {
        if ($node->get_op eq "'") {
            # transpose operator, translate to function
            $result = new Bi::Expression::Function('transpose', [ $node->get_expr ]);
        }
    } elsif ($node->isa('Bi::Expression::BinaryOperator')) {
        if ($node->get_op eq '*') {
            # matrix multiplication, translate to function
            if ($node->get_expr1->is_matrix) {
                if ($node->get_expr2->is_vector) {
                    $result = new Bi::Expression::Function('gemv', [ $node->get_expr1, $node->get_expr2 ]);
                }
            }
        } elsif ($node->get_op =~ /^\.(.*?)$/) {
            # convert element-wise op back to standard scalar ops
            $node->set_op($1);
        }
    }
    
    return $result;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
