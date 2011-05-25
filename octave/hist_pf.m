% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1551 $
% $Date: 2011-05-24 13:50:51 +0800 (Tue, 24 May 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} hist_pf (@var{in}, @var{invar}, @var{coord})
%
% Plot histogram of parameter samples output by pf program.
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
% @end itemize
% @end deftypefn
%
function hist_pf (in, invar, coord)
    % constants
    THRESHOLD = 5e-3; % threshold for bin removal start and end
    BINS = 20;

    % check arguments
    if (nargin < 2 || nargin > 3)
        print_usage ();
    end
    if (nargin < 3)
        coord = [];
    elseif !isvector (coord) || length (coord) > 3
        error ('coord should be a vector with at most three elements');
    end

    % input file
    nci = netcdf (in, 'r');
    T = length (nci ('nr'));
    P = length (nci ('np'));
    
    % read samples
    lws = nci{'logweight'}(T - 1,:);
    ws = exp(lws - max(lws));
    ws = ws / sum(ws);
    x = read_var(nci, invar, coord, [1:P], T - 1);

    ncclose(nci);
    
    % weighted bins
    xmin = min(x);
    xmax = max(x);
    [nn,xx] = hist(x, BINS);
    edges = [xmin, xx(1:end - 1) + diff(xx)/2, xmax + 1];
    [nn,ii] = histc(x, edges);
    
    ww = accumarray(ii', ws);
    
    % plot
    h = bar(xx, ww);
    set(h, 'FaceColor', fade(watercolour(6),0.5), 'EdgeColor', watercolour(6));
end
