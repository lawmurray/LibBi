=head1 NAME

Bi::Visitor::VarReplacer - visitor for replacing variables in expressions.

=head1 SYNOPSIS

    use Bi::Visitor::VarReplacer;
    Bi::Visitor::VarReplacer->evaluate($expr, $from, $to);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::VarReplacer;

use base 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;
use Bi::Utility qw(set_union set_intersect push_unique);

=item B<evaluate>(I<block>, I<from>, I<to>)

Evaluate.

=over 4

=item I<block>

L<Bi::Model::Block> object in which to replace variables.

=item I<from>

Variable, as L<Bi::Model::Var> object, to replace.

=item I<to>

Variable, as L<Bi::Model::Var> object, with which to replace I<from>.

=back

No return value.

=cut
sub evaluate {
    my $class = shift;
    my $block = shift;
    my $from = shift;
    my $to = shift;
    
    assert ($block->isa('Bi::Model::Block')) if DEBUG;
    assert ($from->isa('Bi::Model::Var')) if DEBUG;
    assert ($to->isa('Bi::Model::Var')) if DEBUG;

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
    
    if ($node->isa('Bi::Expression::VarIdentifier')) {
        if ($node->get_var->equals($from)) {
            $node->set_var($to);
        }
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 2867 $ $Date: 2012-07-31 15:38:06 +0800 (Tue, 31 Jul 2012) $
