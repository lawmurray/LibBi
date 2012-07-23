=head1 NAME

Bi::Visitor::InitialToParamTransformer - visitor for augmenting the
parameters of a model with its initial conditions.

=head1 SYNOPSIS

    use Bi::Visitor::InitialToParamTransformer;
    Bi::Visitor::InitialToParamTransformer->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=cut

package Bi::Visitor::InitialToParamTransformer;

use base 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;
use FindBin qw($Bin);
use File::Spec;

use Bi::Visitor::TargetReplacer;

our %MERGE_BLOCKS = (
    'initial' => 'parameter',
    'proposal_initial' => 'proposal_parameter'
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
    
    my $block;
    my $action;
    my $state;
    my $param;
    my $name;
    my $from;
    my $to;
    
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
    
    # create params to hold initial conditions
    foreach $state (@{$model->get_vars('state')}) {
        my @dims = @{$state->get_dims};
        my @args = @{$state->get_args};
        my %named_args = %{$state->get_named_args};
        $named_args{'io'} = new Bi::Expression::Literal(0);
        
        $param = new Bi::Model::Param($state->get_name . "_0_", \@dims,
            \@args, \%named_args);
        $model->add_var($param);
    }
    
    # merge initial blocks into parameter blocks
    foreach $from (keys %MERGE_BLOCKS) {
        if ($model->is_block($from)) {
            my $from_block = $model->get_block($from);
            
            assert ($from_block->num_blocks == 1) if DEBUG;

            # replace state variables with new parameters
            foreach $state (@{$model->get_vars('state')}) {
                $param = $model->get_var($state->get_name . '_0_');
                Bi::Visitor::TargetReplacer->evaluate($from_block, $state, $param);
            }

            # merge blocks
            my $to = $MERGE_BLOCKS{$from};
            $to = $MERGE_BLOCKS{$from};
            if ($model->is_block($to)) {
                my $to_block = $model->get_block($to);
                
                assert ($to_block->num_blocks == 1) if DEBUG;
                
                my $copy_block = $from_block->get_block->clone($model);
                $copy_block->set_commit(1);
                
                $from_block->pop_block;
                $to_block->push_block($copy_block);
                $to_block->sink_children($model);
            } else {
                $from_block->set_name($to);
            }
            
            # replace original with delta mass block
            foreach $state (@{$model->get_vars('state')}) {
                $param = $model->get_var($state->get_name . '_0_');
                $action = new Bi::Model::Action($model->next_action_id,
                    new Bi::Expression::VarIdentifier($state), '<-',
                    new Bi::Expression::VarIdentifier($param));
                $from_block->push_action($action);
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

    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
