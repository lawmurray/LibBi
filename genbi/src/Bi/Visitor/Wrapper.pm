=head1 NAME

Bi::Visitor::Wrapper - visitor for wrapping actions in their preferred parent
block type.

=head1 SYNOPSIS

    use Bi::Visitor::Wrapper;
    Bi::Visitor::Wrapper->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::Wrapper;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<model>)

Evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=back

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = new Bi::Visitor; 
    $self->{_model} = $model;    
    bless $self, $class;
    
    $model->accept($self);
}

=item B<get_model>

Get the model.

=cut
sub get_model {
    my $self = shift;
    return $self->{_model};
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    
    my $result = $node;
    my $subblock;
    my $action;
    my $name;

    my @actions; # actions that remain in this block
    my @optim_blocks; # new sub-blocks to insert
    my @optim_actions; # actions to add to each of these sub-blocks
    my $i;
    
    if ($node->isa('Bi::Model::Block')) {
        ACTION: foreach $action (@{$node->get_actions}) {
            if (!defined($action->get_parent) || $action->get_parent eq $node->get_name) {
                # can stay in this block
                push(@actions, $action);
                next ACTION;
            } elsif ($action->can_combine) {
                # search for another block to add this to
                for ($i = 0; $i < scalar(@optim_blocks); ++$i) {
                    if ($optim_blocks[$i] eq $action->get_parent) {
                        # add to existing block
                        push(@{$optim_actions[$i]}, $action);
                        next ACTION;
                    }
                }
            }

            # add to new block
            push(@optim_blocks, $action->get_parent);
            push(@optim_actions, [ $action ]);
        }
    
        # now actually create all that
        $node->set_actions(\@actions);
        for ($i = 0; $i < @optim_blocks; ++$i) {
            $subblock = new Bi::Model::Block($self->get_model->next_block_id,
                $optim_blocks[$i], [], {}, $optim_actions[$i], []);
               
            push(@{$node->get_blocks}, $subblock);
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
