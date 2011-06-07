% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1410 $
% $Date: 2011-04-15 12:35:42 +0800 (Fri, 15 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} plot_trace2d (@var{in}, @var{invar1}, @var{coord1}, @var{invar2}, @var{coord2}, @var{rang})
%
% Plot trace of samples output by mcmc program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% mcmc.}
%
% @bullet{ @var{invar1} Name of first variable from input file to plot.
%
% @bullet{ @var{invar2} Name of second variable from input file to plot.
%
% @bullet{ @var{coord1} Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar1} to plot.}
%
% @bullet{ @var{coord2} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar2} to plot.}
%
% @bullet{ @var{rang} (optional) Vector of indices of samples to
% plot. All trajectories plotted if not specified.
% @end itemize
% @end deftypefn
%
function plot_trace2d (in, invar1, invar2, coord1, coord2, rang)
    % check arguments
    if nargin < 3 || nargin > 6
        print_usage ();
    end
    if nargin < 4
        coord1 = [];
    end
    if nargin < 5
        coord2 = [];
    end
    if nargin < 6
        rang = [];
    end
    if !check_coord (coord1)
        error ('coord1 should be a vector with at most three elements');
    end
    if !check_coord (coord2)
        error ('coord2 should be a vector with at most three elements');
    end
 
    % input file
    nci = netcdf(in, 'r');

    % data
    P = nci('np')(:);
    if length(rang) == 0
        rang = [1:P];
    end
    
    X1 = read_var (nci, invar1, coord1, rang, 1);
    X2 = read_var (nci, invar2, coord2, rang, 1);
    
    % plot
    ish = ishold;
    plot(X1, X2, 'linewidth', 1, 'color', watercolour(2));
    hold on;
    plot(X1(1), X2(1), '.', 'markersize', 20, 'color', watercolour(2));
    if ish
        hold on
    else
        hold off
    end
  
    %title(nice_name(name, dims));
    %plot_defaults;
    
    ncclose(nci);
end
