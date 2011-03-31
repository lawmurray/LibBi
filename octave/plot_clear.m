% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_clear ()
%
% Clears plot.
% @end deftypefn
%
function plot_clear ()
    % check arguments
    if (nargin > 0)
        print_usage ();
    end

    subplot(1,1,1);
end
