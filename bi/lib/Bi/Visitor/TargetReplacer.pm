=head1 NAME

Bi::Visitor::TargetReplacer - visitor for replacing targets of actions.

=head1 SYNOPSIS

    use Bi::Visitor::TargetReplacer;
    Bi::Visitor::TargetReplacer->evaluate($block, $from, $to);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::TargetReplacer;

use base 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;
use Bi::Utility qw(set_union set_intersect push_unique);

=item B<evaluate>(I<model>)

Evaluate.

=over 4

=item I<block> L<Bi::Model::Block> object.

=back

No return value.

=cut
sub evaluate {
    my $class = shift;
    my $block = shift;
    my $from = shift;
    my $to = shift;

    my $self = new Bi::Visitor;
    bless $self, $class;
    
    $block->accept($self, $from, $to);
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $from = shift;
    my $to = shift;
    
    if ($node->isa('Bi::Model::Action')) {
        if ($node->get_left->get_var->equals($from)) {
            my $ident = new Bi::Expression::VarIdentifier($to,
                $node->get_left->get_indexes);
            $node->set_target($ident);
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
