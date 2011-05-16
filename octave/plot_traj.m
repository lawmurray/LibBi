% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1410 $
% $Date: 2011-04-15 12:35:42 +0800 (Fri, 15 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} plot_simulate (@var{in}, @var{invars}, @var{range}, @var{coord})
%
% Plot trajectories output by simulate, predict, pf, mcmc or likelihood
% program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% simulate.}
%
% @bullet{ @var{invar} Name of variable from input file to plot.
%
% @bullet{ @var{range} (optional) Vector of indices of trajectories to
% plot.
%
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
% plot. All trajectories plotted if not specified.
% @end itemize
% @end deftypefn
%
function plot_traj (in, invar, range, coord)
    % check arguments
    if nargin < 2 || nargin > 4
        print_usage ();
    end
    if nargin < 3
        range = [];
        coord = [];
    elseif nargin < 4
        coord = [];
    elseif !isvector (coord) || length (coord) > 3
        error ('coord should be a vector with at most three elements');
    end
    
    % input file
    nci = netcdf(in, 'r');

    % data
    t = nci{'time'}(:)'; % times
    P = nci('np')(:);
    if length(range) == 0
        range = [1:P];
    end
    
    X = read_var (nci, invar, range, coord);
    
    % plot
    plot(t, X, 'linewidth', 1, 'color', watercolour(1));
    %title(nice_name(name, dims));
    %plot_defaults;
    
    ncclose(nci);
end
