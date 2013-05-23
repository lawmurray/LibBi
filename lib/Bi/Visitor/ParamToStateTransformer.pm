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

use parent 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;

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

    my $parameter_block = $model->get_block('parameter');
    my $initial_block = $model->get_block('initial');
    my $proposal_parameter_block = $model->get_block('proposal_parameter');
    my $proposal_initial_block = $model->get_block('proposal_initial');

    # move the contents of the 'parameter' block to the end of the 'initial'
    # block    
    $initial_block->unshift_children($parameter_block->get_children);
    $parameter_block->clear;
    
    # move the contents of the 'proposal_parameter' block to the end of the
    # 'proposal_initial' block
    $proposal_initial_block->unshift_children($proposal_parameter_block->get_children);    
    $proposal_parameter_block->clear;
    
    # change param variables to state variables    
    $model->accept($self);
}

=item B<visit_after>(I<node>)

Visit node of model.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;

    if ($node->isa('Bi::Model::Var') && $node->get_type eq 'param') {
        $node->set_type('state');
    }
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
