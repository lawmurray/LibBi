=head1 NAME

Bi::Utility - utility functions.

=head1 SYNOPSIS

    use Bi::Utility qw(equals);
    my $list1 = [ $obj1, $obj2 ];
    my $list2 = [ $obj1, $obj2, $obj3 ];
    my $equals = Bi::Utility::equals($list1, $list2);

=head1 METHODS

=over 4

=cut

package Bi::Utility;

use parent 'Exporter';
use warnings;
use strict;

our @EXPORT_OK = qw(equals contains find push_unique unique set_union set_intersect);

use Carp::Assert;

=item B<equals>(I<list1>, I<list2>)

Compare two list references for equality. Elements of each list must
implement an B<equals> function to test for equality.

=cut
sub equals {
    my $list1 = shift;
    my $list2 = shift;
    
    # pre-conditions
    assert(!defined($list1) || ref($list1) eq 'ARRAY') if DEBUG;    
    assert(!defined($list2) || ref($list2) eq 'ARRAY') if DEBUG;    
    
    my $equal = 0;
    my $def1 = defined($list1);
    my $def2 = defined($list2);
    if ($def1 && $def2) {
        my $n1 = scalar(@{$list1});
        my $n2 = scalar(@{$list2});
        $equal = $n1 == $n2;
        
        my $i = 0;
        while ($equal && $i < $n1) {
            $equal = _equals($$list1[$i], $$list2[$i]);
            ++$i;
        }
    }
    return $equal;
}

=item B<contains>(I<list>, I<item>)

Does I<list> contain I<item>?

=cut
sub contains {
    my $list = shift;
    my $item = shift;
    
    # pre-conditions
    assert(defined($list) && ref($list) eq 'ARRAY') if DEBUG;    
    assert(defined($item)) if DEBUG;    

    my $item1;
    foreach $item1 (@$list) {
        if (_equals($item, $item1)) {
            return 1;
        }
    }
    return 0;
}

=item B<find>(I<list>, I<item>)

Returns the index of I<item> in I<list>, or -1 if it does not exist.

=cut
sub find {
    my $list = shift;
    my $item = shift;
    
    # pre-conditions
    assert(defined($list) && ref($list) eq 'ARRAY') if DEBUG;    
    assert(defined($item)) if DEBUG;    

    for (my $i = 0; $i < @$list; ++$i) {
        if (_equals($item, $list->[$i])) {
            return $i;
        }
    }
    return -1;
}

=item B<push_unique>(I<list1>, I<list2/item2>)

Push I<item2> into I<list1>, or all elements of I<list2> into I<list1>, if
they do not already exist as judged by their respective B<equals> functions.

Returns the number of elements added.

=cut
sub push_unique {
    my $list1 = shift;
    my $list2 = shift;
    
    # pre-conditions
    assert(defined($list1) && ref($list1) eq 'ARRAY') if DEBUG;
    if (ref($list2) ne 'ARRAY') {
        $list2 = [ $list2 ];
    }

    my $num = 0;
    my $item;
    foreach $item (@$list2) {
        if (!contains($list1, $item)) {
            push(@$list1, $item);
            ++$num;
        }
    }
    
    return $num;
}

=item B<unique>(I<list>)

Compute new list which removes the duplicates of I<list>.

=cut
sub unique {
    my $list = shift;
    
    my $unique_list = [];
    push_unique($unique_list, $list);
    return $unique_list;
}

=item B<set_union>(I<set1>, I<set2>)

Compute and return union of the two sets.

=cut
sub set_union {
    my $set1 = shift;
    my $set2 = shift;
    
    # pre-conditions
    assert(defined($set1) && ref($set1) eq 'ARRAY') if DEBUG;
    assert(defined($set2) && ref($set2) eq 'ARRAY') if DEBUG;

    my $result = [];
    @$result = @$set1;
    push_unique($result, $set2);
    
    return $result;
}

=item B<set_intersect>(I<set1>, I<set2>)

Compute and return intersection of the two sets.

=cut
sub set_intersect {
    my $set1 = shift;
    my $set2 = shift;
    
    # pre-conditions
    assert(defined($set1) && ref($set1) eq 'ARRAY') if DEBUG;
    assert(defined($set2) && ref($set2) eq 'ARRAY') if DEBUG;

    my $result = [];
    my $item;
    foreach $item (@$set2) {
        if (contains($set1, $item)) {
            push(@$result, $item);
        }
    }
    
    return $result;
}

=item B<_equals>(I<o1>, I<o2>)

Are I<o1> and I<o2> equal?

=cut
sub _equals {
    my $o1 = shift;
    my $o2 = shift;
    
    if (ref($o1) && ref($o2)) {
        return $o1->equals($o2);
    } else {
        return $o1 eq $o2;
    }
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
