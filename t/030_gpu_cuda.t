use Test::More tests => 1;

is(system('script/libbi sample @test.conf --enable-cuda') >> 8, 0, 'GPU with CUDA');
