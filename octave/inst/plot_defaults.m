% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_defaults ()
%
% Sets default decorations and axes for plots.
% @end deftypefn
%
function plot_defaults ()
    % check arguments
    if (nargin > 0)
        print_usage ();
    end

    set(gca, 'interpreter', 'tex');
    set(gca, 'ticklength', [0 0]);
    grid on;
    axis tight;
end
