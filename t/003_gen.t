use Test::More tests => 1;

is(system('libbi sample @test.conf --dry-build --dry-run') >> 8, 0, 'code generation');
