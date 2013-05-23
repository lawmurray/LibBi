=head1 NAME

Bi::Visitor::GetNodesOfType - visitor for constructing list of all nodes
in a hierarchy that match a given type.

=head1 SYNOPSIS

    use Bi::Visitor::GetNodesOfType;
    $vars = Bi::Visitor::GetNodesOfType->evaluate($root,
        ['Bi::Model::Const', 'Bi::Model::Inline']);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::GetNodesOfType;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Bi::Utility qw(contains push_unique);

=item B<new>

Constructor.

=cut
sub new {
    my $class = shift;
    
    my $self = {};
    bless $self, $class;
    
    return $self;
}

=item B<evaluate>(I<node>, I<types>)

Evaluate.

=over 4

=item I<node>

Visitable node.

=item I<types> (optional)

Types of objects to return. If I<types> is given as a string, only
objects of that type are returned. If I<types> is given as an array ref of
strings, only objects of those types are returned. If I<types> is not given,
all objects are returned.

=back

Returns an array ref containing all the unique objects of the given type(s)
in the hierarchy, where uniqueness is assessed using those objects' equality()
method.

=cut
sub evaluate {
    my $class = shift;
    my $node = shift;
    my $types = shift;
    
    if (!defined $types) {
        $types = [];
    } elsif (ref($types) ne 'ARRAY') {
        $types = [ $types ];
    }
    
    my $self = new Bi::Visitor::GetNodesOfType;
    my $nodes = [];
    $node->accept($self, $types, $nodes);
    
    return $nodes;
}

=item B<visit_after>(I<node>)

Visit node.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $types = shift;
    my $nodes = shift;
    
    # recurse through inlines too
    if ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->get_expr->accept($self, $types, $nodes);
    }
    
    foreach my $type (@$types) {
        if ($node->isa($type)) {
            push_unique($nodes, $node);
        }
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
