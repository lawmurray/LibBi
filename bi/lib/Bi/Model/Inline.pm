=head1 NAME

Bi::Model::Inline - inline expression.

=head1 SYNOPSIS

    use Bi::Model::Inline;

=head1 METHODS

=over 4

=cut

package Bi::Model::Inline;

use warnings;
use strict;

use Carp::Assert;
use Bi::Expression;

=item B<new>(I<name>, I<expr>)

Constructor.

=over 4

=item I<name>

Name of the inline expression.

=item I<expr>

The expression, as a L<Bi::Expression> object.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $name = shift;
    my $expr = shift;

    my $self = {
        _name => $name,
        _expr => $expr
    };
    bless $self, $class;
   
    return $self;
}

=item B<get_name>

Get the name of the inline expression.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_expr>

Get the inline expression.

=cut
sub get_expr {
    my $self = shift;
    return $self->{_expr};
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    $self->{_expr} = $self->get_expr->accept($visitor, @args);
    return $visitor->visit($self, @args);
}

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_name eq $obj->get_name;
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
