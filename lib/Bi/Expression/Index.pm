=head1 NAME

Bi::Expression::Index - dimension index in a variable reference.

=head1 SYNOPSIS

    use Bi::Expression::Index;

=head1 INHERITS

L<Bi::Expression>

=head1 DESCRIPTION

An index is a single scalar expression. It is handled internally as a range
with starting expression 0, ending expression 0, size 1, with a dynamic
offset of the index expression.

=head1 METHODS

=over 4

=cut

package Bi::Expression::Index;

use parent 'Bi::Expression';
use warnings;
use strict;

use Bi::Expression::IntegerLiteral;

use Carp::Assert;
use Scalar::Util 'refaddr';

=item B<new>(I<expr>)

Constructor.

=over 4

=item I<expr>

Index expression as L<Bi::Expression> object.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $expr = shift;
    
    assert(defined $expr && $expr->isa('Bi::Expression')) if DEBUG;
    
    my $self = {
        _expr => $expr
    };
    bless $self, $class;

    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = { %$self };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_expr>

Get the index expression. This is used to dynamically offset output.

=cut
sub get_expr {
    my $self = shift;
    return $self->{_expr};
}

=item B<get_start>

Always 0.

=cut
sub get_start {
    my $self = shift;
    return $self->{_expr};
}

=item B<get_end>

Always 0.

=cut
sub get_end {
    my $self = shift;
    return $self->{_expr};
}

=item B<is_index>

Is this an index?

=cut
sub is_index {
    return 1;
}

=item B<is_range>

Is this a range?

=cut
sub is_range {
    return 0;
}

=item B<get_size>

Always 1.

=cut
sub get_size {
    return new Bi::Expression::IntegerLiteral(1);
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

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (
        ref($obj) eq ref($self) &&
        $self->get_expr->equals($obj->get_expr));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

