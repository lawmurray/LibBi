% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_ess (@var{in})
%
% Plot output of the pf program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% pf.}
% @end deftypefn
%
function plot_ess (in)
    % check arguments
    if nargin != 1
        print_usage ();
    end
    
    % input file
    nci = netcdf(in, 'r');

    % data
    lws = nci{'logweight'}(:,:);

    % plot
    plot(ess(lws), 'linewidth', 3, 'color', watercolour(2));
end
