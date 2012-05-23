=head1 NAME

Bi::Visitor::Unroller - visitor for unrolling nested actions from
expressions.

=head1 SYNOPSIS

    use Bi::Visitor::Unroller;
    Bi::Visitor::Unroller->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::Unroller;

use base 'Bi::Visitor';
use warnings;
use strict;

use Bi::Visitor::GetDims;

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

    $model->accept($self, []);
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
    my $args = shift;
    
    my $result = $node;
    my $name;
    my $dims;
    my $class;
    my $tmp;
    my $ident;
    my $action;
    my $arg;
    my $i;

    if ($node->isa('Bi::Expression::Function')) {        
        if (!$node->is_math) {
            # create new action from subexpression
            $action = new Bi::Model::Action($self->get_model->next_action_id,
                    undef, undef, $node);
            if (!$action->can_nest) {
                die("action '" . $action->get_name . "' cannot be nested\n");
            }

            # insert intermediate variable
            $tmp = $self->create_temp_var($node, $node->get_dims);

            # create reference to this new variable
            if ($node->is_element) {
                my @offsets = map { new Bi::Expression::Offset($_, 0) } @{$node->get_aliases};
                $ident = new Bi::Expression::VarIdentifier($tmp, \@offsets);
            } else {
                $ident = new Bi::Expression::VarIdentifier($tmp);
            }
            
            # and unroll
            $action->set_target($ident);
            $result = $ident;
            push(@$args, $action);
        }
    } elsif ($node->isa('Bi::Model::Block') && scalar(@$args)) {
        # construct equivalent of do..then clause for all actions that need
        # to be inserted
        my $block = new Bi::Model::Block($self->get_model->next_block_id);
        my $subblock;
        foreach $action (@$args) {
            $subblock = new Bi::Model::Block($self->get_model->next_block_id,
                    undef, undef, undef, [ $action ]);
            $subblock->set_commit(1);
            $block->push_block($subblock);
        }
        $block->push_block($node);

        @$args = ();
        $result = $block;
    } elsif ($node->isa('Bi::Model::Action')) {
        if ($node->unroll_args) {
            # write arguments to intermediate variables first
            for ($i = 0; $i > $node->num_args; ++$i) {
                $arg = $node->get_args->[$i];
                if (!$arg->is_const && !$arg->isa('Bi::Expression::VarIdentifier')) {
                    # create new action from subexpression
                    $action = new Bi::Model::Action($self->get_model->next_action_id,
                            undef, undef, $arg);
                    
                    # insert intermediate variable
                    $tmp = $self->create_temp_var($arg, $action->get_dims);
                    
                    # create reference to this new variable
                    if ($arg->is_element) {
                        my @offsets = map { new Bi::Expression::Offset($_, 0) } @{$arg->get_aliases};
                        $ident = new Bi::Expression::VarIdentifier($tmp, \@offsets);
                    } else {
                        $ident = new Bi::Expression::VarIdentifier($tmp);
                    }
                
                    # and unroll
                    $action->set_target($ident);
                    $node->get_args->[$i] = $ident;
                    push(@$args, $action);
                }
            }
            foreach $name (keys %{$node->get_named_args}) {
                $arg = $node->get_named_args->{$name};
                if (!$arg->is_const && !$arg->isa('Bi::Expression::VarIdentifier')) {
                    # create new action from subexpression
                    $action = new Bi::Model::Action($self->get_model->next_action_id,
                            undef, undef, $arg);
                    
                    # insert intermediate variable
                    $tmp = $self->create_temp_var($arg, $action->get_dims);
                    
                    # create reference to this new variable
                    if ($arg->is_element) {
                        my @offsets = map { new Bi::Expression::Offset($_, 0) } @{$arg->get_aliases};
                        $ident = new Bi::Expression::VarIdentifier($tmp, \@offsets);
                    } else {
                        $ident = new Bi::Expression::VarIdentifier($tmp);
                    }
                
                    # and unroll
                    $action->set_target($ident);
                    $node->get_named_args->{$name} = $ident;
                    push(@$args, $action);
                }
            }
        }
    }
        
    return $result;
}

sub create_temp_var {
    my $self = shift;
    my $expr = shift;
    my $dims = shift;
    
    my $name = $self->get_model->tmp_var;
    my $named_args = {
        'io' => new Bi::Expression::Literal(0),
        'tmp' => new Bi::Expression::Literal(1)
    };
    my $class;
    
    if ($expr->is_static || $expr->is_common) {
        $class = 'Bi::Model::ParamAux';                       
    } else {
        $class = 'Bi::Model::StateAux';
    }
    my $tmp = $class->new($name, $dims, [], $named_args);
    $self->get_model->add_var($tmp);
            
    return $tmp;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
