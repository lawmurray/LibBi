=head1 NAME

Bi::Model::Target - target of an action.

=head1 SYNOPSIS

    use Bi::Model::Target;

=head1 METHODS

=over 4

=cut

package Bi::Model::Target;

use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<var>, I<aliases>)

Constructor.

=over 4

=item I<var>

Variable referenced, as L<Bi::Model::Var> derived object.

=item I<aliases>

Dimension aliases, as array ref of L<Bi::Model::DimAlias> objects.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $var = shift;
    my $aliases = shift;
    
    # pre-conditions
    assert (defined $var && $var->isa('Bi::Model::Var')) if DEBUG;
    
    assert(!defined($aliases) || ref($aliases) eq 'ARRAY');
    assert(!defined($aliases) || scalar(@$aliases) == 0 || scalar(@$aliases) == $var->num_dims) if DEBUG;
    map { assert($_->isa('Bi::Model::DimAlias')) if DEBUG } @$aliases;

    if (!defined $aliases) {
        $aliases = [];
    }
    
    my $self = {
        _var => $var,
        _aliases => $aliases
    };
    bless $self, $class;
   
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = {
        _var => $self->get_var,
        _aliases => [ map { $_->clone } @{$self->get_aliases} ]
    };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_var>

Get the variable referenced, as L<Bi::Model::Var> derived object.

=cut
sub get_var {
    my $self = shift;
    return $self->{_var};
}

=item B<set_var>(I<var>)

Set the variable referenced.

=cut
sub set_var {
    my $self = shift;
    my $var = shift;
    $self->{_var} = $var;
}

=item B<get_aliases>

Get the dimension aliases, as array ref of L<Bi::Model::DimAlias> objects.

=cut
sub get_aliases {
    my $self = shift;
    return $self->{_aliases};
}

=item B<set_aliases>(I<aliases>)

Set the dimension aliases, as array ref of L<Bi::Model::DimAlias> objects.

=cut
sub set_aliases {
    my $self = shift;
    my $aliases = shift;
    
    # pre-conditions
    assert(!defined($aliases) || ref($aliases) eq 'ARRAY');
    assert(!defined($aliases) || scalar(@$aliases) == 0 || scalar(@$aliases) == $self->get_var->num_dims) if DEBUG;
    map { assert($_->isa('Bi::Model::DimAlias')) if DEBUG } @$aliases;   
    
    $self->{_aliases} = $aliases;
}

=item B<num_aliases>

The number of aliases.

=cut
sub num_aliases {
    my $self = shift;
    return scalar(@{$self->get_aliases});
}

=item B<get_alias>(I<name>)

Look up a dimension alias by name, return undef if doesn't exist, otherwise
L<Bi::Model::DimAlias> object.

=cut
sub get_alias {
	my $self = shift;
	my $name = shift;
	
	foreach my $alias (@{$self->get_aliases}) {
		if ($alias->get_name eq $name) {
			return $alias;
		}
	}
	return undef;
}

=item B<gen_ref>

Create a L<Bi::Expression::VarIdentifier> object corresponding to the target.

=cut
sub gen_ref {
    my $self = shift;
    
    return new Bi::Expression::VarIdentifier($self->get_var, $self->gen_indexes);
}

=item B<gen_indexes>

Create a

=cut
sub gen_indexes {
    my $self = shift;
    
    my $aliases = $self->get_aliases;
    my @indexes = map { $_->gen_index } @$aliases;
    
    return \@indexes;
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    @{$self->{_aliases}} = map { $_->accept($visitor, @args) } @{$self->get_aliases};

    return $visitor->visit($self, @args);
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (
        ref($obj) eq ref($self) &&
        $self->get_var->equals($obj->get_var) &&
        Bi::Utility::equals($self->get_aliases, $obj->get_aliases)); 
}

1;

=back

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 3417 $ $Date: 2013-01-16 15:57:42 +0800 (Wed, 16 Jan 2013) $
