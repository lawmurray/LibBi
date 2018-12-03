
=head1 NAME

Bi::Model - model specification

=head1 SYNOPSIS

    use Bi::Model;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Model;

use parent 'Bi::Block';
use warnings;
use strict;

use Carp::Assert;
use Bi::Model::Const;
use Bi::Model::Inline;
use Bi::Model::Dim;
use Bi::Model::Var;
use Bi::Model::VarGroup;
use Bi::Utility qw(find);

our @BLOCKS = (
    'parameter',
    'initial',
    'transition',
    'observation',
    'bridge'
);

our %MAP_BLOCKS = (
	'proposal_parameter'    => 'parameter',
	'proposal_initial'      => 'initial',
	'lookahead_transition'  => 'transition',
	'lookahead_observation' => 'observation'
);

=item B<new>

Constructor.

=cut

sub new {
	my $class = shift;

	my $self = new Bi::Block;
	$self->{_name} = undef;

	bless $self, $class;
	
	# add built-in variables
	$self->push_var(new Bi::Model::Var('builtin_', 't_now'));
	$self->push_var(new Bi::Model::Var('builtin_', 't_last_input'));
	$self->push_var(new Bi::Model::Var('builtin_', 't_next_obs'));
	
	return $self;
}

=item B<clone>

Return a clone of the object.

=cut

sub clone {
	my $self = shift;

	my $clone = Bi::Block::clone($self);
	$clone->{_name} = $self->get_name;

	bless $clone, ref($self);
	return $clone;
}

=item B<get_name>

Get the name of the model.

=cut

sub get_name {
	my $self = shift;

	return $self->{_name};
}

=item B<set_name>(I<name>)

Set the name of the model

=cut

sub set_name {
	my $self = shift;
	my $name = shift;

	$self->{_name} = $name;
}

=item B<get_size>(I<type>)

Get the size of the model. If I<type> is given, the size of the net for that
particular type is returned, otherwise the sum of all sizes is returned.

=cut

sub get_size {
	my $self = shift;
	my $type;
	if (@_) {
		$type = shift;
	}

	my $vars = $self->get_all_vars($type);
	my $var;
	my $size = 0;
	foreach $var (@$vars) {
		$size += $var->get_size;
	}

	return $size;
}

=item B<get_dim_id>(I<dim>)

Get the id of a dimension I<dim>. This is used to assign ids to dimensions
when generating code, ensuring that ids start from zero and that none are
skipped. These guarantees are not provided by L<Bi::Model::Dim::get_id>.

=cut

sub get_dim_id {
	my $self = shift;
	my $dim  = shift;
	
	return find($self->get_all_dims, $dim);
}

=item B<get_var_id>(I<var>, I<types>)

Get the id of a variable I<var> among those of any type given in the array
ref I<types>. If I<types> is not given, the variable's own type only is used.
The order of types is important, as ids are assigned to all variables of the
first type, then all variables of the second type, etc, in sequence.

This is used, among other places, to assign ids to variables when generating
code, ensuring that ids start from zero for each variable type, and that none
are skipped. These guarantees are not provided by L<Bi::Model::Var::get_id>.

=cut

sub get_var_id {
	my $self  = shift;
	my $var   = shift;
	my $types = shift;

	if (!defined $types) {
		$types = [ $var->get_type ];
	}

	my $result = 0;
    TYPE: foreach my $type (@$types) {
		my $vars = $self->get_all_vars($type);
		if ($var->get_type eq $type) {
			$result = find($vars, $var);
			last TYPE;
		} else {
			$result += scalar(@$vars);
		}
	}
	return $result;
}

=item B<get_var_start>(I<var>)

Get the starting index of variable I<var> among those of its type. This is
the sum of the sizes of all preceding variables.

=cut
sub get_var_start {
	my $self = shift;
	my $var  = shift;

	my $type = $var->get_type;
	my $var2;
	my $start = 0;
	foreach $var2 (@{$self->get_all_vars($type)}) {
		if ($var->equals($var2)) {
			last;
		} else {
			$start += $var2->get_size;
		}
	}

	return $start;
}

=item B<get_var_group_start>(I<group>)

Get the starting index of variable group I<group> among those of its type.
This is simply the starting index of the first variable in the group, or
zero if the group is empty.

=cut
sub get_var_group_start {
	my $self = shift;
	my $group  = shift;
	
    my $vars = $group->get_vars;
    if (@$vars) {
        return $self->get_var_start($vars->[0]);
    } else {
        return 0;
    }
}

=item B<get_pair_var>(I<prefix>, I<var1>, I<var2>)

Get the variable named "prefix_var1name_var2name_". This is used mainly
by L<Bi::Visitor::ExtendedTransformer>.

=cut

sub get_pair_var {
	my $self   = shift;
	my $prefix = shift;
	my $var1   = shift;
	my $var2   = shift;

	my $name =
	  sprintf("%s_%s_%s_", $prefix, $var1->get_name, $var2->get_name);

	return $self->get_var($name);
}

=item B<add_pair_var>(I<prefix>, I<var1>, I<var2>)

Create the variable named "prefix_var1name_var2name_". This is used mainly
by L<Bi::Visitor::ExtendedTransformer>.

=cut

sub add_pair_var {
	my $self   = shift;
	my $prefix = shift;
	my $var1   = shift;
	my $var2   = shift;

	my $name =
	  sprintf('%s_%s_%s_', $prefix, $var1->get_name, $var2->get_name);
	my $dims = [ @{ $var1->get_dims }, @{ $var2->get_dims } ];
	my $named_args = {
		'has_input'  => new Bi::Expression::IntegerLiteral(0),
		'has_output' => new Bi::Expression::IntegerLiteral(0)
	};
	my $j_var =
	  new Bi::Model::Var('state_aux_', $name, $dims, [], $named_args);

	return $j_var;
}

=item B<add_column_var>(I<prefix>, I<rows>, I<var2>)

Create the variable named "prefix_var2name_" with an opening dimension of
length I<rows>. This is used mainly by L<Bi::Visitor::ExtendedTransformer>.

=cut

sub add_column_var {
	my $self   = shift;
	my $prefix = shift;
	my $rows   = shift;
	my $var2   = shift;

	my $name = sprintf('%s_%s_', $prefix, $var2->get_name);
	my $row_dim = $self->lookup_dim($rows);
	my $dims = [ $row_dim, @{ $var2->get_dims } ];

	my $named_args = {
		'has_input'  => new Bi::Expression::IntegerLiteral(0),
		'has_output' => new Bi::Expression::IntegerLiteral(0)
	};
	my $j_var =
	  new Bi::Model::Var('state_aux_', $name, $dims, [], $named_args);

	return $j_var;
}

=item B<get_jacobian_var>(I<var1>, I<var2>)

Get the variable named "F_var1name_var2name_" or "G_var1name_var2name",
depending on the variable types. This is used mainly by
L<Bi::Visitor::ExtendedTransformer>.

=cut

sub get_jacobian_var {
	my $self = shift;
	my $var1 = shift;
	my $var2 = shift;

	my $prefix = ($var2->get_type eq 'obs') ? 'G' : 'F';

	return $self->get_pair_var($prefix, $var1, $var2);
}

=item B<add_jacobian_var>(I<var1>, I<var2>)

Create the variable named "F_var1name_var2name_" or "G_var1name_var2name",
depending on the variable types. This is used mainly by
L<Bi::Visitor::ExtendedTransformer>.

=cut

sub add_jacobian_var {
	my $self = shift;
	my $var1 = shift;
	my $var2 = shift;

	my $prefix = ($var2->get_type eq 'obs') ? 'G' : 'F';

	return $self->add_pair_var($prefix, $var1, $var2);
}

=item B<get_std_var>(I<var1>, I<var2>)

Get the variable named "Q_var1name_var2name_" or "R_var1name_var2name",
depending on the variable types. This is used mainly by
L<Bi::Visitor::ExtendedTransformer>.

=cut

sub get_std_var {
	my $self = shift;
	my $var1 = shift;
	my $var2 = shift;

	my $prefix = ($var2->get_type eq 'obs') ? 'R' : 'Q';

	return $self->get_pair_var($prefix, $var1, $var2);
}

=item B<add_std_var>(I<var1>, I<var2>)

Create the variable named "Q_var1name_var2name_" or "R_var1name_var2name",
depending on the variable types. This is used mainly by
L<Bi::Visitor::ExtendedTransformer>.

=cut

sub add_std_var {
	my $self = shift;
	my $var1 = shift;
	my $var2 = shift;

	my $prefix = ($var2->get_type eq 'obs') ? 'R' : 'Q';

	return $self->add_pair_var($prefix, $var1, $var2);
}

=item B<validate>

Validate model.

=cut

sub validate {
	my $self = shift;

	my $name;
	my $block;

	# complete top-level blocks
	foreach $name (@BLOCKS) {
	    if (!$self->is_block($name)) {
	        $block = new Bi::Block;
	        $block->set_name($name);
	        $block->validate;
	        $self->push_child($block);
	    }
	}
	foreach $name (sort keys %MAP_BLOCKS) {
		if (!$self->is_block($name)) {
			$block = $self->get_block($MAP_BLOCKS{$name})->clone;
			$block->set_name($name);
			$block->validate;
			$self->push_child($block);
		}
	}
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut

sub accept {
	my $self    = shift;
	my $visitor = shift;
	my @args    = @_;

	Bi::Block::accept($self, $visitor, @args);
}

=item B<equals>(I<obj>)

=cut

sub equals {
	my $self = shift;
	my $obj  = shift;

	return ref($obj) eq ref($self) && $self->get_name eq $obj->get_name;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

