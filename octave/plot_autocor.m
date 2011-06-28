% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_autocor (@var{in}, @var{invar}, @var{coord}, @var{rang})
%
% Plot autocorrelation of samples output by mcmc program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% mcmc.}
%
% @bullet{ @var{invar} Name of variable from input file to plot.
%
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
%
% @bullet{ @var{rang} (optional) Vector of indices of samples to
% plot. All trajectories plotted if not specified.
% @end itemize
% @end deftypefn
%
function plot_autocor (in, invar, coord, rang)
    % check arguments
    if nargin < 2 || nargin > 4
        print_usage ();
    end
    if nargin < 3
        coord = [];
        rang = [];
    elseif nargin < 4
        rang = [];
    end
    if !check_coord (coord)
        error ('coord should be a vector with at most three elements');
    end
  
    % input file
    nci = netcdf(in, 'r');

    % data
    P = nci('np')(:);
    if length(rang) == 0
        rang = [1:P];
    end
    
    X = read_var (nci, invar, coord, rang);
    
    % plot
    plot(autocor(X), 'linewidth', 1, 'color', watercolour(2));
    %title(nice_name(name, dims));
    %plot_defaults;
    
    ncclose(nci);
end
