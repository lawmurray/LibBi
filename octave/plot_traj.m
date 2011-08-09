% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_traj (@var{in}, @var{invar}, @var{coord}, @var{ps}, @var{ts})
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
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
% plot. All trajectories plotted if not specified.
%
% @bullet{ @var{ps} (optional) Trajectory indices.
%
% @bullet{ @var{ts} (optional) Time indices.
% @end itemize
% @end deftypefn
%
function plot_traj (in, invar, coord, ps, ts)
    % check arguments
    if nargin < 2 || nargin > 5
        print_usage ();
    end
    if nargin < 3
        coord = [];
    elseif !check_coord (coord)
        error ('coord should be a vector with at most three elements');
    end
    if nargin < 4
        ps = [];
    end
    if nargin < 5
        ts = [];
    end
  
    % input file
    nci = netcdf(in, 'r');

    % data
    P = length (nci('np'));
    T = length (nci('nr'));
    if isempty (ps)
        ps = [1:P];
    end
    if isempty (ts)
        ts = [1:T];
    end
    
    t = nci{'time'}(ts)'; % times
    X = read_var (nci, invar, coord, ps, ts);
    
    % plot
    plot(t, X, 'linewidth', 1, 'color', gray()(32,:));
    %title(nice_name(name, dims));
    %plot_defaults;
    
    ncclose(nci);
end
