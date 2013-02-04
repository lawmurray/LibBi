=head1 NAME

Bi::Model::Dim - dimension.

=head1 SYNOPSIS

    use Bi::Model::Dim;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Model::Dim;

use base 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;

our $DIM_ARGS = [
  {
    name => 'size',
    positional => 1,
    mandatory => 1
  },
  {
    name => 'boundary',
    positional => 1,
    default => "'none'"
  }
];

=item B<new>(I<name>, I<args>, I<named_args>)

Constructor.

=over 4

=item I<name>

Name of the dimension.

=item I<args>

Ordered list of positional arguments, as L<Bi::Expression> objects.

=item I<named_args>

Hash of named arguments, keyed by name, as L<Bi::Expression> objects.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $name = shift;
    my $args = shift;
    my $named_args = shift;

    my $self = new Bi::ArgHandler($args, $named_args);
    $self->{_id} = -1;
    $self->{_name} = $name;

    bless $self, $class;

    $self->validate;
   
    return $self;
}

1;

=item B<get_id>

Get the id of the dimension (-1 until assigned to a L<Bi::Model>).

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<set_id>

Set the id of the dimension.

=over 4

=item I<id>

Id of the dimension.

=back

=cut
sub set_id {
    my $self = shift;
    my $id = shift;
    
    $self->{_id} = $id;
}

=item B<get_name>

Get the name of the dimension.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_size>

Get the size of the dimension.

=cut
sub get_size {
    my $self = shift;
    return $self->get_named_arg('size')->eval_const;
}

=item B<validate>

Validate dimension.

=cut
sub validate {
    my $self = shift;
    
    $self->process_args($DIM_ARGS);
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    Bi::ArgHandler::accept($self, $visitor, @args);
    return $visitor->visit($self, @args);
}

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_name eq $obj->get_name;
}

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
