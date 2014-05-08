=head1 NAME

Bi::Expression::Range - dimension range in a variable reference.

=head1 SYNOPSIS

    use Bi::Expression::Range;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Range;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

=item B<new>(I<start>, I<end>)

Constructor.

=over 4

=item I<start>

L<Bi::Expression> giving starting index.

=item I<end>

L<Bi::Expression> giving ending index.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $start = shift;
    my $end = shift;
    
    assert(defined $start && $start->isa('Bi::Expression')) if DEBUG;
    assert(defined $end && $end->isa('Bi::Expression')) if DEBUG;
    
    my $self = {
        _start => $start,
        _end => $end,
    };
    bless $self, $class;

    return $self;
}

=item B<is_index>

Is this an index?

=cut
sub is_index {
    return 0;
}

=item B<is_range>

Is this a range?

=cut
sub is_range {
    return 1;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = {
    	_start => $self->has_start ? $self->get_start->clone : undef,
    	_end => $self->has_end ? $self->get_end->clone : undef
    };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_start>

Get the starting index.

=cut
sub get_start {
    my $self = shift;
    return $self->{_start};
}

=item B<has_start>

Is there a starting index?

=cut
sub has_start {
    my $self = shift;
    return defined $self->{_start};
}

=item B<get_end>

Get the ending index.

=cut
sub get_end {
    my $self = shift;
    return $self->{_end};
}

=item B<has_end>

Is there an ending index?

=cut
sub has_end {
    my $self = shift;
    return defined $self->{_end};
}

=item B<get_size>

Size of the range.

=cut
sub get_size {
    my $self = shift;
    
    if ($self->has_start && $self->has_end) {
        return $self->get_end - $self->get_start + 1;
    } else {
        return [];
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
	    if ($self->has_start) {
    	    $self->{_start} = $self->get_start->accept($visitor, @args);
    	}
    	if ($self->has_end) {
        	$self->{_end} = $self->get_end->accept($visitor, @args);
    	}
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
        $self->has_start == $obj->has_start &&
        (!$self->has_start || $self->get_start->equals($obj->get_start)) &&
        $self->has_end == $obj->has_end &&
        (!$self->has_end || $self->get_end->equals($obj->get_end)));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
