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

=item * C<--help> Display general help, or help for client program (if client
program is specified).

=item * C<--verbose> Enable verbose reporting.

=item * C<--dry-build> Generate code, but do not build or execute.

=item * C<--dry-run> Generate code and build, but do not execute.

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
use Bi::Visitor::InitialToParamTransformer;

use Carp;
use Carp::Assert;
use Getopt::Long qw(:config pass_through no_auto_abbrev);
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
        'dry-run!' => \$self->{_dry_run},
        'dry-build!' => \$self->{_dry_build},
        'model-file=s' => \$self->{_model_file},
        );
    GetOptions(@args) || die("could not read command line arguments\n");
    
    # error and warning handlers
    if ($self->{_verbose}) {
        $SIG{__DIE__} = sub { Carp::confess(@_) };
        $SIG{__WARN__} = sub { Carp::cluck(@_) };
    }
    
    # command
    $self->{_cmd} = shift @ARGV;

    return $self;
}

=item B<do>

Perform action according to command line.

=cut
sub do {
    my $self = shift;
    
    my $cmd = $self->{_cmd};
    eval {
        if ($self->{_help}) {
           $self->help;
        } elsif (!defined($cmd) || $cmd eq '') {
            die("no command given\n");
        } elsif ($cmd eq 'setup') {
              $self->setup;
        } elsif ($cmd eq 'clean') {
            $self->clean;
        } elsif ($cmd eq 'draw') {
            $self->draw;
        } elsif ($cmd eq 'document') {
        	$self->document;
        } else {
            $self->client;
        }
    };
    if ($@) {
        _error($@);
    }
}

=item B<help>

Display help.

=cut
sub help {
    my $self = shift;    

    if (defined $self->{_cmd}) {
        # command-specific help
        print "Command specific help\n";
    } else {
        # general help
        print "General help\n";
    }
}

=item B<draw>

Draw graph of model.

=cut
sub draw {
    my $self = shift;
    
    if (!defined $self->{_model_file}) {
        die("no model specified\n");
    } else {
        my $fh = new IO::File;
        $fh->open($self->{_model_file}) || die("could not open " . $self->{_model_file} . "\n");
        my $parser = new Bi::Parser();
        my $model = $parser->parse($fh);
        $fh->close;    
    
        my $optimiser = new Bi::Optimiser($model);
        $optimiser->optimise();

        my $outdir = '.' . $model->get_name;   
        my $dot = new Bi::Gen::Dot($outdir);
        $dot->gen($model);
    }
}

=item B<document>

Write Doxygen configuration file and generate HTML documentation.

=cut
sub document {
    my $self = shift;
    
    if (!defined $self->{_model_file}) {
        die("no model specified\n");
    } else {
        my $fh = new IO::File;
        $fh->open($self->{_model_file}) || die("could not open " . $self->{_model_file} . "\n");
        my $parser = new Bi::Parser();
        my $model = $parser->parse($fh);
        $fh->close;
        
        my $outdir = '.' . $model->get_name;   
        my $doxyfile = new Bi::Gen::Doxyfile($outdir);
        $doxyfile->gen($model);
        
        chdir($outdir);
        exec('doxygen') || die("exec failed ($!)\n");;
    }
}

=item B<client>

Build client program (if not C<--dry-build>) and run (if not C<--dry-run>).

=cut
sub client {
    my $self = shift;
    my $cmd = $self->{_cmd};
    
    if (!defined $self->{_model_file}) {
        die("no model specified\n");
    }
    
    $self->_report("Parsing...");
    my $fh = new IO::File;
    $fh->open($self->{_model_file}) || die("could not open " . $self->{_model_file} . "\n");
    my $parser = new Bi::Parser();
    my $model = $parser->parse($fh);
    $fh->close;

    my $outdir = '.' . $model->get_name;

    my $optimiser = new Bi::Optimiser($model);
    my $cpp = new Bi::Gen::Cpp($outdir);
    my $build = new Bi::Gen::Build($outdir);
    my $builder = new Bi::Builder($outdir, $self->{_verbose});    
    my $client = new Bi::Client($cmd, $outdir, $self->{_verbose});    

    $self->_report("Processing arguments...");
    $client->process_args;
    
    $self->_report("Transforming model...");
    if ($client->get_named_arg('transform-param-to-state')) {
        Bi::Visitor::ParamToStateTransformer->evaluate($model);
    } elsif ($client->get_named_arg('transform-initial-to-param')) {
        Bi::Visitor::InitialToParamTransformer->evaluate($model);
    }
    if ($client->get_named_arg('transform-extended')) {
        Bi::Visitor::ExtendedTransformer->evaluate($model);
    }

    $self->_report("Optimising model...");
    $optimiser->optimise();

    if ($client->is_cpp) {
        $self->_report("Generating code...");
        $cpp->gen($model, $client);
        
        $self->_report("Generating build system...");
        $build->gen($model);

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
    my $msg = shift;
            
    # tidy up message
    chomp $msg;
    if ($msg !~ /^Error/) {
        $msg = "Error: $msg";
    }

    die("$msg\n");
}

=item B<_warn>(I<msg>)

Print I<msg>.

=cut
sub _warn {
    my $msg = shift;
    
    # tidy up message
    chomp $msg;
    if ($msg !~ /^Warning/) {
        $msg = "Warning: $msg";
    }
    
    warn("$msg\n");
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
