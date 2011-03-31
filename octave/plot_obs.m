% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_obs (@var{in}, @var{invars})
%
% Plot observations.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file containing
% observations.}
%
% @bullet{ @var{invars} Cell array of strings naming the variables
% of this file to plot. Empty strings may be used to produce empty plots.}
% @end itemize
% @end deftypefn
%
function plot_obs (in, invars, ns)
    % check arguments
    if (nargin < 2 || nargin > 3)
        print_usage ();
    end
    if (nargin == 3)
        if (!isscalar(ns))
            error ('ns must be scalar');
        end
        if (ns < 1)
            error ('ns must be positive');
        end
    end
    if (nargin == 2)
        ns = 1;
    end 

    % input file
    nci = netcdf(in, 'r');

    t = nci{'time'}(:)'; % times
    for i = 1:length(invars)
        if (!strcmp(invars{i}, ''))
            y = nci{invars{i}}(ns,:);
            
            subplot(length(invars), 1, i);
            hold on;
            plot(t, y, 'ok');
        end
    end
    
    ncclose(nci);
end
