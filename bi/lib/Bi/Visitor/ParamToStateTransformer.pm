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

    my $parameter_block = $model->get_block('parameter')->clone($model);
    $parameter_block->set_name('eval');
    $parameter_block->set_commit(1);
    
    my $proposal_parameter_block = $model->get_block('proposal_parameter')->clone($model);
    $proposal_parameter_block->set_name('eval');
    $proposal_parameter_block->set_commit(1);
    
    $model->accept($self, $model, $parameter_block, $proposal_parameter_block);
}

=item B<visit>(I<node>)

Visit node of model.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $model = shift;    
    my $parameter_block = shift;
    my $proposal_parameter_block = shift;

    if ($node->isa('Bi::Model::Param')) {
        # replace with state variable
        $node = new Bi::Model::State($node->get_name, $node->get_dims,
            $node->get_args, $node->get_named_args);
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
    	if ($node->get_var->get_type eq 'param') {
    		my $name = $node->get_var->get_name;
    		$node->set_var($model->get_var($name));
    	} 
    } elsif ($node->isa('Bi::Model::Block')) {
        if ($node->get_name eq 'parameter' || $node->get_name eq 'proposal_parameter') {
            $node->clear;
        } elsif ($node->get_name eq 'initial') {
            $node->sink_actions($model);
            $node->unshift_block($parameter_block);
        } elsif ($node->get_name eq 'proposal_initial') {
            $node->sink_actions($model);
            $node->unshift_block($proposal_parameter_block);
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
