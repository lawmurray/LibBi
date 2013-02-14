=head1 NAME

Bi::Visitor::Resolver - visitor for resolving read/write conflicts in model.

=head1 SYNOPSIS

    use Bi::Visitor::Resolver;
    Bi::Visitor::Resolver->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::Resolver;

use base 'Bi::Visitor';
use warnings;
use strict;

use Bi::Visitor::TargetReplacer;

use Carp::Assert;
use Bi::Utility qw(set_union set_intersect push_unique);

=item B<evaluate>(I<model>)

Evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=back

No return value.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = new Bi::Visitor;
    $self->{_model} = $model;
    bless $self, $class;
    
    $model->accept($self, [], [], [], [], [], []);
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
    my $reads = shift;
    my $writes = shift;
    my $inplace_reads = shift;
    my $inplace_writes = shift;
    my $conflicts = shift;

    my $sub_reads = [];
    my $sub_writes = [];
    my $sub_inplace_reads = [];
    my $sub_inplace_writes = [];
    my $sub_conflicts = [];

    my $ref;
    my $set;
    my $name;
    my $var;
    my $action;
    my $dims;
    my $tmp;

    if ($node->isa('Bi::Model::Action')) {
        if ($node->is_inplace) {
            $sub_inplace_reads = [ map { $_->get_var } @{$node->get_vars} ];
            $sub_inplace_writes = [ $node->get_left->get_var ];
        } else {
            $sub_reads = [ map { $_->get_var } @{$node->get_vars} ];
            $sub_writes = [ $node->get_left->get_var ];
            
            # an action only conflicts with itself if it is not an inplace
            # action, and reads from the same variable to which it writes,
            # with nontrivial indexes along the dimensions of that variable
            foreach $ref (@{$node->get_vars}) {
                if ($ref->get_var->equals($node->get_left->get_var) &&
                        !$ref->trivial_index) {
                    push_unique($sub_conflicts, $node->get_left->get_var);
                    last;
                }
            }
            
            # ...or if it marks itself as such
            if ($node->unroll_target) {
                push_unique($sub_conflicts, $node->get_left->get_var);
            }
        }

        $set = set_intersect(set_union($writes, $inplace_writes),
                set_union($sub_writes, $sub_inplace_writes));
        map { die("conflicting writes to variable '" . $_->get_name .
               "', resolve with do..then clause'\n") } @$set;

        $set = set_intersect($reads, $inplace_writes);
        map { die("conflicting read and inplace write of variable '" .
               $_->get_name . "', resolve with do..then clause\n") } @$set;

        push_unique($conflicts, set_intersect($reads, $sub_writes));
        push_unique($conflicts, set_intersect($writes, $sub_reads));
        push_unique($conflicts, set_intersect($inplace_reads, $sub_writes));
        push_unique($conflicts, set_intersect($writes, $sub_inplace_reads));

        push_unique($conflicts, $sub_conflicts);
        push_unique($reads, $sub_reads);
        push_unique($writes, $sub_writes);
        push_unique($inplace_reads, $sub_inplace_reads);
        push_unique($inplace_writes, $sub_inplace_writes);
    } elsif ($node->isa('Bi::Model::Block')) {
        if ($node->get_commit) {
            if (@$conflicts) {
                # replace with two blocks, the first preloading conflicting
                # variables into temporaries, the second as the original
                # block but replacing reads of conflicted variables with
                # reads from these temporaries.
                my $read_block = new Bi::Model::Block(
                    $self->get_model->next_block_id);
                my $write_block = new Bi::Model::Block(
                    $self->get_model->next_block_id);

                $write_block->set_blocks($node->get_blocks);
                $write_block->set_actions($node->get_actions);
                $node->set_blocks([ $read_block, $write_block ]);
                $node->set_actions([]);

                foreach $var (@$conflicts) {
                    # insert intermediate output variable
                    $name = $self->get_model->tmp_var;
                    $dims = $var->get_dims;
                    $tmp = ref($var)->new($name, $dims, [],
                        {
                            'has_input' => new Bi::Expression::IntegerLiteral(0),
                            'has_output' => new Bi::Expression::IntegerLiteral(0)
                        });
                    $self->get_model->add_var($tmp);
        
                    # update reads from this variable in write block
                    Bi::Visitor::VarReplacer->evaluate($node, $var, $tmp);

                    # add preload to read block
                    $action = new Bi::Model::Action(
                           $self->get_model->next_action_id,
                           new Bi::Model::Target($tmp),
                           '<-',
                           new Bi::Expression::VarIdentifier($var));

                    $read_block->push_action($action);
                }
            }

            # conflicts now resolved
            @$reads = ();
            @$writes = ();
            @$inplace_reads = ();
            @$inplace_writes = ();
            @$conflicts = ();
        }
        
        # post-condition
        assert(!$node->get_commit || !scalar(@$conflicts)) if DEBUG;
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
