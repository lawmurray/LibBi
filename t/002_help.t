use Test::More tests => 1;

is(system('script/libbi help >/dev/null') >> 8, 0, 'help system');
