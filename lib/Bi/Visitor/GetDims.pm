=head1 NAME

Bi::Visitor::GetDims - visitor for constructing list of dimensions along
which the result of an expression is defined.

=head1 SYNOPSIS

    use Bi::Visitor::GetDims;
    $dims = Bi::Visitor::GetDims->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::GetDims;

use parent 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=back

Returns an array ref containing a list of L<Bi::Model::Dim> objects giving
the dimensions along which the result of the expression extends.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    my $self = new Bi::Visitor;
    bless $self, $class;
    
    my $dims = [];
    $expr->accept($self, $dims);
    
    return $dims;
}

=item B<visit_after>(I<node>)

Visit node of expression tree.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $dims = shift;
    
    my $sub_dims = [];
    if ($node->isa('Bi::Expression::VarIdentifier')) {
    	if (@{$node->get_indexes} == 0) {
	        $sub_dims = $node->get_var->get_dims;
    	    @$dims = @$sub_dims;
    	}
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->accept($self, $dims);
    } elsif ($node->isa('Bi::Action')) {
        $sub_dims = $node->get_dims;
        @$dims = @$sub_dims;
    }
    
    # TODO: only check for basic operators
    #if (@$sub_dims) {
    #    if (@$dims) {
    #        # check that number of dimensions matches
    #        if (scalar(@$sub_dims) != scalar(@$dims)) {
    #            die("incompatible dimension counts in expression\n");
    #        }
    #               
    #        # check that dimension sizes match
    #        my $i;
    #        for ($i = 0; $i < @$dims; ++$i) {
    #            if ($dims->[$i]->get_size != $sub_dims->[$i]->get_size) {
    #                die("incompatible dimension sizes in expression\n");
    #            }
    #        }
    #    } else {
    #        @$dims = @$sub_dims;
    #    }
    #}
                
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

