=head1 NAME

draw - visualise a model specification as a directed graph.

=head1 SYNOPSIS

    bi draw --model I<model.bi> > I<model.dot>
    dot -Tpdf -o I<model.pdf> I<model.dot>

=head1 DESCRIPTION

The C<draw> command takes a model specification and outputs a directed graph
that visualises the model. It is useful for validation and debugging
purposes. The output is in the format of a DOT script. It will need to be
processed by the C<dot> program in order to create an image (see example
above).

=head1 OPTIONS

The following options are supported:

=over 4

=item * C<--model-file> the model specification file.

=back
