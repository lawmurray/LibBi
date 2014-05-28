use Test::More tests => 3;

is(system('make --version >/dev/null') >> 8, 0, 'make presence');
is(system('automake --version >/dev/null') >> 8, 0, 'automake presence');
is(system('autoconf --version >/dev/null') >> 8, 0, 'autoconf presence');
