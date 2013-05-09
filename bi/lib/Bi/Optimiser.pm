=head1 NAME

Bi::Optimiser - optimise model.

=head1 SYNOPSIS

    use Bi::Optimiser;
    my $optimiser = new Bi::Optimiser($model);
    $optimiser->optimise;

=head1 DESCRIPTION

I<Bi::Optimiser> transforms a model to an internal form required for
compilation. As part of this it applies a number of transformations to
optimise performance.

=head1 METHODS

=over 4

=cut

package Bi::Optimiser;

use warnings;
use strict;

use Carp::Assert;

use Bi::Model;
use Bi::Visitor::Simplify;
use Bi::Visitor::Unroller;
use Bi::Visitor::Wrapper;
use Bi::Visitor::StaticExtractor;
use Bi::Visitor::StaticReplacer;
    
=item B<new>(I<model>)

Constructor.

=over 4

=item I<model> The model to opimise.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $model = shift;
    
    my $self = {
        _model => $model
    };
    bless $self, $class;
    return $self;
}

=item B<optimise>

Apply all optimisations to the model.

=cut
sub optimise {
    my $self = shift;

    my $model = $self->{_model};
        
    my ($lefts, $rights) = Bi::Visitor::StaticExtractor->evaluate($model);    
    Bi::Visitor::StaticReplacer->evaluate($model, $lefts, $rights);
    
    Bi::Visitor::Unroller->evaluate($model);
    Bi::Visitor::Wrapper->evaluate($model);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
