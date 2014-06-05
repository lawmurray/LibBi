use Test::More tests => 1;

is(system('script/libbi sample @test.conf --enable-sse') >> 8, 0, 'CPU with SSE');
