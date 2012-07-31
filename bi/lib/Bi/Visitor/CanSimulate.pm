=head1 NAME

Bi::Visitor::CanSimulate - can block be simulated?

=head1 SYNOPSIS

    use Bi::Visitor::CanSimulate;
    
    if (Bi::Visitor::CanSimulate->evaluate($expr)) {
        ...
    }

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::CanSimulate;

use base 'Bi::Visitor';
use warnings;
use strict;

use List::Util;

=item B<evaluate>(I<block>)

Evaluate.

=over 4

=item I<block> Block.

=back

Returns true if I<block> contains only actions using the '<-' operator, false
otherwise.

=cut
sub evaluate {
    my $class = shift;
    my $block = shift;
    
    my $self = new Bi::Visitor; 
    bless $self, $class;
    
    my $args = [ 1 ];
    $block->accept($self, $args);
    
    return $args->[0];
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $args = shift;
    
    if ($node->isa('Bi::Model::Action')) {
        $args->[0] = int($args->[0] && $node->get_op eq '<-');
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
