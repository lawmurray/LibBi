#!/usr/bin/perl

=head1 NAME

Bi::FrontEnd - frontend to bi functionality.

=head1 SYNOPSIS

    use Bi::FrontEnd;
    my $frontend = new Bi::FrontEnd;
    $frontend->do;

=head1 COMMAND LINE

C<Bi::FrontEnd> is controlled from the command line. The command line should
take the form:

    bi I<command> I<options>

or:

    bi I<options>

where C<command> is the command to run. The following options are
supported at this level, additional arguments depend on the client program:

=over 4

=item C<--help> Display general help.

=item C<--verbose> Enable verbose reporting.

=item C<--dry-build> Generate code, but do not build or execute.

=item C<--dry-run> Generate code and build, but do not execute.

=back

=head1 METHODS

=over 4

=cut

package Bi::FrontEnd;

use warnings;
use strict;

use Bi::Builder;
use Bi::Parser;
use Bi::Model;
use Bi::Gen::Dot;
use Bi::Gen::Cpp;
use Bi::Gen::Build;
use Bi::Gen::Doxyfile;
use Bi::Client;
use Bi::Optimiser;
use Bi::Visitor::ExtendedTransformer;
use Bi::Visitor::ParamToStateTransformer;
use Bi::Visitor::ObsToStateTransformer;
use Bi::Visitor::InitialToParamTransformer;

use Carp;
use Carp::Assert;
use Getopt::Long qw(:config pass_through no_auto_abbrev no_ignore_case);
use File::Path;
use IO::File;

=item B<new>

Constructor.

=cut
sub new {
    my $class = shift;

    my $self = {
        _help => 0,
        _verbose => 0,
        _dry_gen => 0,
        _dry_run => 0,
        _dry_build => 0,
        _cmd => undef,
        _model_file => undef
    };
    bless $self, $class;
    
    # command line options
    my @args = (
        'help' => \$self->{_help},
        'verbose!' => \$self->{_verbose},
        'dry-gen!' => \$self->{_dry_gen},
        'dry-run!' => \$self->{_dry_run},
        'dry-build!' => \$self->{_dry_build},
        'model-file=s' => \$self->{_model_file},
        );
    GetOptions(@args) || die("could not read command line arguments\n");
    
    # error and warning handlers
  	$SIG{__DIE__} = sub { $self->_error(@_) };
   	$SIG{__WARN__} = sub { $self->_warn(@_) };
    
    # command
    $self->{_cmd} = shift @ARGV;

    return $self;
}

=item B<do>

Perform action according to command line.

=cut
sub do {
    my $self = shift;
    
    eval {
        if ($self->{_help}) {
            unshift(@ARGV, $self->{_cmd});
            $self->{_cmd} = 'help';
        }
        if (!defined($self->{_cmd}) || $self->{_cmd} eq '') {
            die("no client given\n");
        } else {
            $self->client;
        }
    };
    if ($@) {
        die($@);
    }
}

=item B<client>

Build client program (if not C<--dry-build>) and run (if not C<--dry-run>).

=cut
sub client {
    my $self = shift;
    my $cmd = $self->{_cmd};
    
    # parse
    my $model = undef;
    if (defined $self->{_model_file}) {
        $self->_report("Parsing...");
        my $fh = new IO::File;
        $fh->open($self->{_model_file}) || die("could not open " . $self->{_model_file} . "\n");
        my $parser = new Bi::Parser;
        $model = $parser->parse($fh);
        $fh->close;
        if ($model->get_name . '.bi' ne $self->{_model_file}) {
            warn("model name does not match model file name\n");
        }
    }
    
    # generators etc
    my $dirname = (defined $model) ? $model->get_name : 'Bi';
    my $builder = new Bi::Builder($dirname, $self->{_verbose});
    my $cpp = new Bi::Gen::Cpp($builder->get_dir);
    my $build = new Bi::Gen::Build($builder->get_dir);
    my $client = new Bi::Client($cmd, $builder->get_dir, $self->{_verbose});    

    # process args
    if (!defined $model && $client->needs_model) {
        die("no model specified\n");
    }
    $self->_report("Processing arguments...");
    $client->process_args;

    # transform
    if (defined $model && $client->needs_transform) {
        $self->_report("Transforming model...");
        if ($client->get_named_arg('with-transform-param-to-state')) {
            Bi::Visitor::ParamToStateTransformer->evaluate($model);
        } elsif ($client->get_named_arg('with-transform-initial-to-param')) {
            Bi::Visitor::InitialToParamTransformer->evaluate($model);
        }
        if ($client->get_named_arg('with-transform-obs-to-state')) {
            Bi::Visitor::ObsToStateTransformer->evaluate($model);
        }
        if ($client->get_named_arg('with-transform-extended')) {
            Bi::Visitor::ExtendedTransformer->evaluate($model);
        }
    
        # optimise
        my $optimiser = new Bi::Optimiser($model);
        $optimiser->optimise;
    
        # doxygen
        my $doxyfile = new Bi::Gen::Doxyfile($builder->get_dir);
        $doxyfile->gen($model);
    }

    # generate code and build
    if ($client->is_cpp) {
        if (!$self->{_dry_gen}) {
            $self->_report("Generating code...");
            $cpp->gen($model, $client);
            
            $self->_report("Generating build system...");
            $build->gen($model);
        }
        if (!$self->{_dry_build}) {
            $self->_report("Building...");
            $builder->build($client->get_binary);
        }
    }
        
    if (!$self->{_dry_run}) {
        $self->_report("Running...");
        $client->exec($model);
    }
}

=item B<_report>(I<msg>)

Print I<msg>, if C<--verbose>.

=cut
sub _report {
    my $self = shift;
    my $msg = shift;
    
    if ($self->{_verbose}) {
        print STDERR "$msg\n";
    }
}

=back

=head1 CLASS METHODS

=over 4

=item B<_error>(I<msg>)

Print I<msg> and exit.

=cut
sub _error {
	my $self = shift;
    my $msg = shift;
            
    # tidy up message
    chomp $msg;
    if ($msg !~ /^Error/) {
        $msg = "Error: $msg";
    }

	if ($self->{_verbose}) {
		Carp::confess("$msg\n");
	} else {
		die("$msg\n");
	}
}

=item B<_warn>(I<msg>)

Print I<msg>.

=cut
sub _warn {
	my $self = shift;
    my $msg = shift;
    
    # tidy up message
    chomp $msg;
    if ($msg !~ /^Warning/) {
        $msg = "Warning: $msg";
    }
    
	if ($self->{_verbose}) {
		Carp::cluck("$msg\n");
	} else {
		warn("$msg\n");
	}
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
