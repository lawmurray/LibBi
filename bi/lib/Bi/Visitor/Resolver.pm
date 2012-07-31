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

    my $result = $node;
    my $sub_reads = [];
    my $sub_writes = [];
    my $sub_inplace_reads = [];
    my $sub_inplace_writes = [];
    my $sub_conflicts = [];

    my $ref;
    my $set;
    my $name;
    my $var;
    my $commit_block;
    my $action;
    my $dims;
    my $tmp;

    if ($node->isa('Bi::Model::Action')) {
        if ($node->is_inplace) {
            $sub_inplace_reads = [ map { $_->get_var } @{$node->get_vars} ];
            $sub_inplace_writes = [ $node->get_target->get_var ];
        } else {
            $sub_reads = [ map { $_->get_var } @{$node->get_vars} ];
            $sub_writes = [ $node->get_target->get_var ];
            
            # an action only conflicts with itself if it is not an inplace
            # action, and reads from the same variable to which it writes,
            # with nonzero offsets along the dimensions of that variable
            foreach $ref (@{$node->get_vars}) {
                if ($ref->get_var->equals($node->get_target->get_var) &&
                        !$ref->no_offset) {
                    push_unique($sub_conflicts, $node->get_var);
                    last;
                }
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
                # replace with two blocks, one that does computation and
                # writes to intermediate variables, one that does commit
                $commit_block = Bi::Model::Block->new_copy_block(
                        $self->get_model->next_block_id);
                $result = new Bi::Model::Block($self->get_model->next_block_id,
                        undef, [], {}, [], [ $node, $commit_block ]);

                foreach $var (@$conflicts) {
                    # insert intermediate output variable
                    $name = $self->get_model->tmp_var;
                    $dims = $var->get_dims;
                    $tmp = ref($var)->new($name, $dims, [],
                        {
                            'io' => new Bi::Expression::Literal(0),
                            'tmp' => new Bi::Expression::Literal(1)
                        });
                    $self->get_model->add_var($tmp);
        
                    # update references to this variable in children
                    Bi::Visitor::TargetReplacer->evaluate($node, $var, $tmp);
                        
                    # add copy back to commit block
                    $action = Bi::Model::Action->new_copy_action(
                           $self->get_model->next_action_id,
                           new Bi::Expression::VarIdentifier($var),
                           new Bi::Expression::VarIdentifier($tmp));

                    $commit_block->push_action($action);
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
    
    return $result;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
