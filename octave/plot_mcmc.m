% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_mcmc (@var{ins}, @var{invar}, @var{coord}, @var{ps}, @var{ts})
%
% Plot output of the mcmc program.
%
% @itemize
% @bullet{ @var{ins} Input file.s Gives the names of NetCDF files output by
% mcmc.}
%
% @bullet{ @var{invar} Name of variable from input file to plot.
%
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
% @end itemize
% @end deftypefn
%
function plot_mcmc (ins, invar, coord, ps, ts)
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
    if iscell (ins) && !iscellstr (ins)
        error ('ins must be a string or cell array of strings');
    elseif ischar (ins)
        ins = { ins };
    end

    X = [];
    for i = 1:length (ins)
        % input file
        in = ins{i};
        nci = netcdf(in, 'r');
        T = length (nci('nr'));
        if isempty (ts)
            ts = [1:T];
        end
        t = nci{'time'}(ts)'; % times
        x = read_var (nci, invar, coord, ps, ts);
        X = [ X x ];
        
        ncclose (nci);
    end

    % data
    q = [0.025 0.5 0.975]'; % quantiles (median and 95%)
    Q = quantile (X, q, 2);
    
    % plot
    ish = ishold;
    if !ish
        cla; % patch doesn't clear otherwise
    end
    area_between(t, Q(:,1), Q(:,3), watercolour(2));
    hold on;
    plot(t, Q(:,2), 'linewidth', 3, 'color', watercolour(2));
    if ish
        hold on;
    else
        hold off;
    end
    %title(nice_name(name, dims));
    %plot_defaults;
    
    ncclose(nci);
end
