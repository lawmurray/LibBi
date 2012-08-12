=head1 NAME

Bi::Jacobian - utility functions for Jacobian manipulations.

=head1 SYNOPSIS

    use Bi::Jacobian;

=head1 METHODS

=over 4

=cut

package Bi::Jacobian;

use base 'Exporter';
use warnings;
use strict;

use Carp::Assert;

=item B<commit>(I<model>, I<vars>, I<J>)

=cut
sub commit {
    my $model = shift;
    my $vars = shift;
    my $J = shift;
    
    my $nonzeros = new Bi::Expression::Matrix($J->num_rows, $J->num_cols);
    for (my $i = 0; $i < @$vars; ++$i) {
        for (my $j = 0; $j < @$vars; ++$j) {
            my $expr = $J->get($i, $j);
            if (defined($expr)) {
            	if ($expr->is_one) {
            		$nonzeros->set($i, $j, $expr->clone);
            	} else {
	                my $var1 = $vars->[$i];
	                my $var2 = $vars->[$j];
	                my $J_var = $model->get_jacobian_var($var1, $var2); 
	                my $ref = new Bi::Expression::VarIdentifier($J_var);
	                
	                $nonzeros->set($i, $j, $ref);
            	}
            }
        }
    }
    return $nonzeros;
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
