=head1 NAME

help - look up online help for an action, block or command.

=head1 SYNOPSIS

    bi help I<name>
    
    bi help I<name> --action
    
    bi help I<name> --block
    
    bi help I<name> --command

=head1 DESCRIPTION

The C<help> command is used to access documentation from the command line.
This documentation is the same as that provided in the user reference.

C<I<name>> gives the name of any action, block or command. The documentation
for the respective action, block or command is presented. Ambiguities (e.g.
actions and blocks with the same name) are resolved via a prompt, or by
using any of the C<--action>, C<--block> or C<--command> options.

=cut

package Bi::Client::help;

use parent 'Bi::Client';
use warnings;
use strict;

use Pod::Usage;
use Pod::Find qw(pod_where);

=head1 OPTIONS

The following options are supported:

=over 4

=item C<--action>

explicitly search for an action.

=item C<--block>

explicitly search for a block.

=item C<--command>

explicitly search for a command.

=back

=cut

our @CLIENT_OPTIONS = (
    {
      name => 'action',
      type => 'bool'
    },
    {
      name => 'block',
      type => 'bool'
    },
    {
      name => 'command',
      type => 'bool'
    }
);
  
=head1 METHODS

=over 4

=cut

sub init {
    my $self = shift;

    $self->{_name} = shift @ARGV;
    $self->{_binary} = undef;
    
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub get_name {
    my $self = shift;
    return $self->{_name};
}

sub is_cpp {
    return 0;
}

sub needs_model {
    return 0;
}

sub exec {
    my $self = shift;
    my $model = shift;
    
    my $name = $self->get_name;
    
    if (defined $name) {    
        # include all types...
        my $include_action = 1;
        my $include_block = 1;
        my $include_command = 1;
        
        # ...unless there is one or more explicit qualifiers
        if ($self->is_named_arg('action') || $self->is_named_arg('block') ||
            $self->is_named_arg('command')) {
            $include_action = $self->is_named_arg('action');
            $include_block = $self->is_named_arg('block');
            $include_command = $self->is_named_arg('command');
        }
        
        # search
        my $have_action = 0;
        my $have_block = 0;
        my $have_command = 0;
        my $class;

        if ($include_action) {
            $have_action = 1;
            my $action = "Bi::Action::$name";
            if (eval("require $action")) {
                $class = $action;
            }
        }
        if ($include_block) {
            $have_block = 1;
            my $block = "Bi::Block::$name";
            if (eval("require $block")) {
                $class = $block;
            }
        }
        if ($include_command) {
            $have_command = 1;
            my $command = "Bi::Client::$name";
            if (eval("require $command")) {
                $class = $command;
            }
        }

        if (defined $class) {
            pod2usage({
                -exitval => 0,
                #-verbose => 2,  # perldoc style
                -verbose => 99,
                -sections => 'NAME|SYNOPSIS|DESCRIPTION|OPTIONS|PARAMETERS|AUTHOR|VERSION',
                -input => pod_where({ -inc => 1 }, $class)
            });
        } else {
            die("cannot find '$name'\n");
        }
    } else {
        $self->_general_help;
    }
}

sub _general_help {
    my $self = shift;
        print <<End;
Usage: bi <command> [options]

where <command> is one of:
  * draw
  * filter
  * help
  * optimise
  * package
  * rewrite
  * sample
  * simulate

Type 'bi help <command>' for help on a particular command. For more information
on using the help command type 'bi help help'.
End
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
