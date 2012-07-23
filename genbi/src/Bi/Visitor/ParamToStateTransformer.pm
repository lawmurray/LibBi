=head1 NAME

Bi::Visitor::ParamToStateTransformer - visitor for augmenting the state of a
model with its parameters.

=head1 SYNOPSIS

    use Bi::Visitor::ParamToStateTransformer;
    Bi::Visitor::ParamToStateTransformer->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=cut

package Bi::Visitor::ParamToStateTransformer;

use base 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;
use FindBin qw($Bin);
use File::Spec;

our %MERGE_BLOCKS = (
    'parameter' => 'initial',
    'proposal_parameter' => 'proposal_initial'
);

=head1 METHODS

=over 4

=item B<evaluate>(I<model>)

Construct and evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=back

No return value.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = {};
    bless $self, $class;

    my $name;
    my $block;
    
    if ($model->is_block('proposal_parameter') !=
        $model->is_block('proposal_initial')) {
        # explicitly create remaining proposal blocks to ensure consistent
        # behaviour when merging, recalling that proposal_parameter defers to
        # parameter, and proposal_initial defers to initial, if they do not
        # exist.
        foreach $name ('initial', 'parameter') {
            if (!$model->is_block("proposal_$name")) {
                $block = $model->get_block($name)->clone($model);
                $block->set_name("proposal_$name");
                $model->add_block($block);        
            }
        }
    }

    # move parameters to state
    $model->accept($self);
    
    # merge parameter blocks into initial blocks
    foreach my $from (keys %MERGE_BLOCKS) {
        if ($model->is_block($from)) {
            my $from_block = $model->get_block($from);
            
            assert ($from_block->num_blocks == 1) if DEBUG;

            my $to = $MERGE_BLOCKS{$from};
            if ($model->is_block($to)) {
                my $to_block = $model->get_block($to);
                
                assert ($to_block->num_blocks == 1) if DEBUG;
                
                my $copy_block = $from_block->get_block->clone($model);
                $copy_block->set_commit(1);
                
                $from_block->shift_block;
                $to_block->unshift_block($copy_block);
                $to_block->sink_children($model);
            } else {
                $from_block->set_name($to);
            }
        }
    }    
}

=item B<visit>(I<node>)

Visit node of model.

=cut
sub visit {
    my $self = shift;
    my $node = shift;

    if ($node->isa('Bi::Model::Param')) {
        # replace with state
        $node = new Bi::Model::State($node->get_name, $node->get_dims,
            $node->get_args, $node->get_named_args);
    }
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
