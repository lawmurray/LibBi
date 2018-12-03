=head1 NAME

Bi::Gen::Doxyfile - generator for Doxygen configuration files.

=head1 SYNOPSIS

    use Bi::Gen::Doxyfile;
    $gen = new Bi::Gen::Doxyfile($outdir);
    $gen->gen($model);

=head1 INHERITS

L<Bi::Gen>

=head1 METHODS

=over 4

=cut

package Bi::Gen::Doxyfile;

use parent 'Bi::Gen';
use warnings;
use strict;

use Carp::Assert;
use File::Spec;

use Bi qw(share_file share_dir);
use Bi::Gen;
use Bi::Model;
use Bi::Expression;
use Bi::Visitor::ToAscii;

=item B<new>(I<outdir>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $outdir = shift;

    my $ttdir = share_dir(File::Spec->catdir('tt', 'doxyfile'));
    my $self = Bi::Gen->new($ttdir, $outdir);

    bless $self, $class;    
    return $self;
}

=item B<gen>(I<model>)

Generate code for model.

=cut

sub gen {
    my $self = shift;
    my $model = shift;

    # pre-condition
    assert(!defined $model || $model->isa('Bi::Model')) if DEBUG;

    if (defined $model) {
	    $self->process_templates('model', { 'model' => $model });
    }
}

=item B<process_templates>(I<template_name>, I<vars>, I<output_name>)

Process all template files which have the given name, plus a file extension
of C<.bi.tt>.

=cut
sub process_templates {
    my $self = shift;
    my $template_name = shift;
    my $vars = shift;
    
    $self->process_template("Doxyfile.tt", $vars, 'Doxyfile');
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

