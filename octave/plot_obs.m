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
% @bullet{ @var{invar} Name of variable from input file to plot.
%
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
%
% @bullet{ @var{ns} (optional) Index along ns dimension of input file.}
% @end itemize
% @end deftypefn
%
function plot_obs (in, invar, coord, ns)
    % check arguments
    if nargin < 2 || nargin > 4
        print_usage ();
    end
    if nargin == 2
        coord = [];
        ns = 1;
    elseif nargin == 3
        ns = 1;
    else
        if (!isscalar(ns))
            error ('ns must be scalar');
        end
        if (ns < 1)
            error ('ns must be positive');
        end
    end

    % input file
    nci = netcdf(in, 'r');

    % time variable
    tvar = sprintf('time_%s', invar);
    K = length(nci{tvar}(:));
    ts = nci{tvar}(:);

    % coordinate variable
    cvar = sprintf('coord_%s', invar);
    if length(coord) > 1
        coords = nci{cvar}(:,:);
    else
        coords = nci{cvar}(:);
    end
    
    % observed variables
    ys = nci{invar}(ns,:);
    
    % mask based on coordinates
    if isempty(coords)
        mask = ones(K,1);
    else
        mask = zeros(K,1);
        for k = 1:K
            if coords(k,:) == coord
                mask(k) = 1;
            end
        end
    end
    t = ts(find(mask));
    y = ys(find(mask));
    
    % plot
    plot(t, y, 'ok', 'markersize', 3.0);
    %plot_defaults;
    
    ncclose(nci);
end
