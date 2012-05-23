=head1 NAME

Bi::Model::Const - constant.

=head1 SYNOPSIS

    use Bi::Model::Const;

=head1 METHODS

=over 4

=cut

package Bi::Model::Const;

use warnings;
use strict;

use Carp::Assert;
use Bi::Expression;

=item B<new>(I<name>, I<expr>)

Constructor.

=over 4

=item I<name>

Name of the constant.

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

    my $self = {
        _name => $name,
        _expr => $expr
    };
    bless $self, $class;
   
    $self->validate;
   
    return $self;
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

    $self->{_expr} = $self->get_expr->accept($visitor, @args);
    return $visitor->visit($self, @args);
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
