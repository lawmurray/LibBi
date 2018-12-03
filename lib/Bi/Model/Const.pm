=head1 NAME

Bi::Model::Const - constant.

=head1 SYNOPSIS

    use Bi::Model::Const;

=head1 METHODS

=over 4

=cut

package Bi::Model::Const;

use parent 'Bi::Node';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

use Bi::Expression;

our $_next_const_id = 0;

=item B<new>(I<name>, I<expr>)

Constructor.

=over 4

=item I<name>

Name of the constant. If undefined a unique name is generated.

=item I<expr>

Value of the constant as a L<Bi::Expression> object. Must not contain
references to variables.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $name = shift;
    my $expr = shift;

    my $id = $_next_const_id++;
    my $self = {
        _id => $id,
        _name => (defined $name) ? $name : sprintf("const_%d_", $id),
        _expr => $expr
    };
    bless $self, $class;
   
    $self->validate;
   
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = {
        _id => $_next_const_id++,
        _name => $self->get_name,
        _expr => $self->get_expr->clone
    };
    
    bless $clone, ref($self);
    return $clone;
}

=item B<get_id>

Get the id of the constant.

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<get_name>

Get the name of the constant.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_expr>

Get the expression of the constant.

=cut
sub get_expr {
    my $self = shift;
    return $self->{_expr};
}

=item B<validate>

Validate constant.

=cut
sub validate {
    my $self = shift;
    
    if (!$self->get_expr->is_const) {
        die("expression for 'const' may only depend on literals and other constants\n");
    }
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    my $new = $visitor->visit_before($self, @args);
    if (refaddr($new) == refaddr($self)) {
        $self->{_expr} = $self->get_expr->accept($visitor, @args);
        $new = $visitor->visit_after($self, @args);
    }
    return $new;
}

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_id == $obj->get_id;
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

