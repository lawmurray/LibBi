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

use parent 'Bi::Visitor';
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
    bless $self, $class;

    $model->accept($self, $model, []);
}

=item B<visit_after>(I<node>)

Visit node.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $model = shift;
    my $actions = shift;
    
    my @results;
    
    if ($node->isa('Bi::Expression::Function')) {        
        if (!$node->is_math) {
            # can't unroll element expressions
            if ($node->is_element) {
                die("cannot unroll action '" . $node->get_name . "' with element expressions\n");
            }
            my $action = $self->_unroll_expr($model, $node);
            $node = $action->get_left->clone;
            push(@$actions, $action->accept($self, $model, $actions));
        }
        push(@results, $node);
    } elsif ($node->isa('Bi::Action')) {
        if ($node->unroll_args) {
            # write arguments to intermediate variables first
            for (my $i = 0; $i < $node->num_args; ++$i) {
                my $arg = $node->get_args->[$i];
                if (!$arg->is_const && !$arg->is_basic) {
                    my $action = $self->_unroll_expr($model, $arg);
                    $node->get_args->[$i] = $action->get_left->clone;
                    push(@$actions, $action);
                }
            }
            foreach my $name (sort keys %{$node->get_named_args}) {
                my $arg = $node->get_named_args->{$name};
                if (!$arg->is_const && !$arg->is_basic) {
                    my $action = $self->_unroll_expr($model, $arg);
                    $node->get_named_args->{$name} = $action->get_left->clone;
                    push(@$actions, $action);
                }
            }
        }
        push(@$actions, $node);
        push(@results, @$actions);
        @$actions = ();
    } else {
        push(@results, $node);
    }

    return @results[0..$#results];
    # ^ not sure why the slice is necessary, but 'return @results' doesn't
    # work, something to do with list vs scalar context perhaps?
}

=item B<_unroll_expr>(I<model>, I<expr>)

=cut
sub _unroll_expr {
    my $self = shift;
    my $model = shift;
    my $expr = shift;

    # temporary variable to hold expression result
    my $type = ($expr->is_common) ? 'param_aux_' : 'state_aux_';
    my $dims = [ map { $model->lookup_dim($_) } @{$expr->get_shape->get_sizes} ];
    
  	my $var = new Bi::Model::Var($type, undef, $dims, [], {
        'has_input' => new Bi::Expression::IntegerLiteral(0),
        'has_output' => new Bi::Expression::IntegerLiteral(0)
    });
    $model->push_var($var);

    # action to evaluate expression
    my $left = new Bi::Expression::VarIdentifier($var, $var->gen_ranges);
    
    my $action = new Bi::Action;
    $action->set_aliases($var->gen_aliases);
    $action->set_left($left);
    $action->set_op('<-');
    $action->set_right($expr);
    $action->validate;
    
    if (!$action->can_nest) {
        die("action '" . $action->get_name . "' cannot be nested, it must appear on a line of its own\n");
    }
    
    return $action;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
