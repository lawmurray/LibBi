use Test::More tests => 1;

is(system('libbi help >/dev/null') >> 8, 0, 'help system');
