=head1 NAME

Bi::Model::Input - input variable.

=head1 SYNOPSIS

    use Bi::Model::Input;

=head1 INHERITS

L<Bi::Model::Var>

=head1 METHODS

=over 4

=cut

package Bi::Model::Input;

use base 'Bi::Model::Var';
use warnings;
use strict;

use Bi::Model::Var;

=item B<new>(I<name>, I<dims>)

Constructor.

=cut
sub new {
    my $class = shift;
    
    my $self = Bi::Model::Var->new(@_);
    $self->{_type} = 'input';
    bless $self, $class;
    return $self;
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
