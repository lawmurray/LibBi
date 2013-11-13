=head1 NAME

bridge - the bridge potential.

=head1 SYNOPSIS

    sub bridge {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<bridge> block may reference variables of any type, but may
only target variables of type C<noise> and C<state>. References to C<obs>
variables provide their next value. Use of the built-in variables C<t_now>
and C<t_next_obs> will be useful.

=cut

package Bi::Block::bridge;

use parent 'Bi::Block::observation';
use warnings;
use strict;

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
