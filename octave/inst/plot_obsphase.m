% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_obs (@var{in}, @var{invar1}, @var{invar2}, @var{coord}, @var{ns})
%
% Plot observations in phase space.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file containing
% observations.}
%
% @bullet{ @var{invar1} Name of variable from input file for x-axis.
%
% @bullet{ @var{invar2} Name of variable from input file for y-axis.
%
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
%
% @bullet{ @var{ns} (optional) Index along ns dimension of input file.}
% @end itemize
% @end deftypefn
%
function plot_obsphase (in, invar1, invar2, coord, ns)
    % check arguments
    if nargin < 3 || nargin > 5
        print_usage ();
    end
    if nargin < 4
        coord = [];
    end
    if nargin < 5
        ns = 1;
    else
        if !isscalar(ns)
            error ('ns must be scalar');
        end
        if ns < 1
            error ('ns must be positive');
        end
    end

    % input file
    nci = netcdf(in, 'r');

    % read
    [t1 y1] = read_obs (nci, invar1, coord, ns);
    [t2 y2] = read_obs (nci, invar2, coord, ns);    
    if !isequal (t1, t2)
        error (sprintf(['Observations %s and %s not coincident for '
            'phase plot'], invar1, invar2));
    end
    
    % plot
    plot(y1, y2, 'ok', 'markersize', 3.0);
    %plot_defaults;
    
    ncclose(nci);
end
