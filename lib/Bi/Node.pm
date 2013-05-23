=head1 NAME

Bi::Node - node of a model hierarchy.

=head1 SYNOPSIS

    use Bi::Node;

=head1 METHODS

=over 4

=cut

package Bi::Node;

use warnings;
use strict;

use Carp::Assert;
use Bi::Visitor::GetNodesOfType;
use Bi::Utility qw(contains push_unique);

=item B<new>

Constructor.

=cut
sub new {
    my $class = shift;
    
    my $self = {};
    bless $self, $class;
    
    return $self;
}

=item B<get_all_const_refs>

Get all constants referenced in the hierarchy, as an array ref of
L<Bi::Expression::ConstIdentifier> objects.

=cut
sub get_all_const_refs {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Expression::ConstIdentifier');
}

=item B<get_all_inline_refs>

Get all inlines referenced in the hierarchy, as an array ref of
L<Bi::Expression::InlineIdentifier> objects.

=cut
sub get_all_inline_refs {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Expression::InlineIdentifier');
}

=item B<get_all_alias_ref>

Get all dimension aliases used in the hierarchy, as an array ref of
L<Bi::Expression::DimAliasIdentifier> objects.

=cut
sub get_all_alias_refs {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Expression::DimAliasIdentifier');
}

=item B<get_all_var_refs>(I<types>)

Get all variables referenced in the hierarchy, as an array ref of
L<Bi::Expression::VarIdentifier> objects. If I<types> is given as a string,
only variables of that type are returned. If I<types> is given as an array ref
of strings, only variables of those types are returned. If I<types> is not
given, all variables are returned.

=cut
sub get_all_var_refs {
    my $self = shift;
    my $types = shift;
    
    my $refs = Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Expression::VarIdentifier');
    if (defined $types) {
        if (ref($types) ne 'ARRAY') {
            $types = [ $types ];
        }
        my @refs = map { contains($types, $_->get_var->get_type) ? $_ : () } @$refs;
        $refs = \@refs;
    }
    return $refs;
}

=item B<get_all_left_var_refs>

Get all variable references that appear on the left hand side of actions.

=cut
sub get_all_left_var_refs {
    my $node = shift;
    
    my $results = [];
    if ($node->isa('Bi::Action')) {
        push_unique($results, $node->get_left);
    } elsif ($node->isa('Bi::Block')) {
        foreach my $child (@{$node->get_children}) {
            push_unique($results, $child->get_all_left_var_refs);
        }
    }
    return $results;
}

=item B<get_all_right_var_refs>

Get all variable references that appear on the right hand side of actions.

=cut
sub get_all_right_var_refs {
    my $node = shift;
    
    my $results = [];
    if ($node->isa('Bi::Action')) {
        push_unique($results, $node->get_right_var_refs);
    } elsif ($node->isa('Bi::Block')) {
        foreach my $child (@{$node->get_children}) {
            push_unique($results, $child->get_all_right_var_refs);
        }
    }
    return $results;
}

=item B<get_all_consts>

Get all constants declared in the hierarchy, as an array ref of
L<Bi::Model::Const> objects.

=cut
sub get_all_consts {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Model::Const');
}

=item B<get_all_inlines>

Get all inlines declared in the hierarchy, as an array ref of
L<Bi::Model::Inline> objects.

=cut
sub get_all_inlines {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Model::Inline');
}

=item B<get_all_dims>

Get all dimensions declared in the hierarchy, as an array ref of
L<Bi::Model::Dim> objects.

=cut
sub get_all_dims {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Model::Dim');
}

=item B<get_all_aliases>

Get all dimension aliases declared in the hierarchy, as an array ref of
L<Bi::Model::DimAlias> objects.

=cut
sub get_all_aliases {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Model::DimAlias');
}

=item B<get_all_vars>(I<types>)

Get all variables referenced in the hierarchy, as an array ref of
L<Bi::Model::Var> objects. If I<types> is given as a string,
only variables of that type are returned. If I<types> is given as an array ref
of strings, only variables of those types are returned. If I<types> is not
given, all variables are returned.

If I<types> is not given, variables are returned in the order declared. If
I<types> is given, variables are returned sorted by type in the order of
I<types>, and then by the order declared.

=cut
sub get_all_vars {
    my $self = shift;
    my $types = shift;

    my $vars = Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Model::Var');
    my $results;
    
    if (!defined $types) {
        $results = $vars;
    } else {
        if (ref($types) ne 'ARRAY') {
            $types = [ $types ];
        }
        foreach my $type (@$types) {
            push(@$results, map { ($_->get_type eq $type) ? $_ : () } @$vars);
        }
    }
    return $results;
}

=item B<get_all_left_vars>

Get all variables that appear on the left hand side of actions.

=cut
sub get_all_left_vars {
    my $node = shift;
    
    my $results = [];
    if ($node->isa('Bi::Action')) {
        push_unique($results, $node->get_left->get_var);
    } elsif ($node->isa('Bi::Block')) {
        foreach my $child (@{$node->get_children}) {
            push_unique($results, $child->get_all_left_vars);
        }
    }
    return $results;
}

=item B<get_all_right_vars>

Get all variables that appear on the right hand side of actions.

=cut
sub get_all_right_vars {
    my $node = shift;
    
    my $results = [];
    if ($node->isa('Bi::Action')) {
        push_unique($results, $node->get_right_vars);
    } elsif ($node->isa('Bi::Block')) {
        foreach my $child (@{$node->get_children}) {
            push_unique($results, $child->get_all_right_vars);
        }
    }
    return $results;
}

=item B<get_all_blocks>

Get all blocks declared in the hierarchy, as an array ref of
L<Bi::Block> objects.

=cut
sub get_all_blocks {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Block');
}

=item B<get_all_actions>

Get all actions declared in the hierarchy, as an array ref of
L<Bi::Action> objects.

=cut
sub get_all_actions {
    my $self = shift;
    
    return Bi::Visitor::GetNodesOfType->evaluate($self, 'Bi::Action');
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
