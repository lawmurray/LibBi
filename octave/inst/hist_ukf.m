% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} hist_ukf (@var{in}, @var{invar}, @var{logn})
%
% Plot histogram of parameter samples output by ukf program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% ukf.}
%
% @bullet{ @var{invar} Name of variable from input file to plot.
%
% @bullet{ @var{logn} (optional) True if this is a log-variable, false
% otherwise.
% @end itemize
% @end deftypefn
%
function hist_ukf (in, invar, logn)
    % constants
    THRESHOLD = 5e-3; % threshold for bin removal start and end
    BINS = 20;

    % check arguments
    if (nargin < 2 || nargin > 3)
        print_usage ();
    elseif (nargin < 3)
        logn = 0;
    end

    % input file
    nci = netcdf(in, 'r');

    % read
    id = nci{invar}(end) + 1;
    mu = nci{'filter.mu'}(end,id);
    sigma = sqrt(nci{'filter.Sigma'}(end,id,id));
            
    % construct curve
    xmin = mu - 3*sigma;
    xmax = mu + 3*sigma;
    if logn
        xmin = exp(xmin);
        xmax = exp(xmax);
    end
    x = xmin:(xmax - xmin)/100:xmax;
    if logn
        y = lognpdf(x, mu, sigma);
    else
        y = normpdf(x, mu, sigma);
    end
    
    % plot
    plot(x, y, 'linewidth', 3, 'color', watercolour(3));
    %title(invar);
    %hist_defaults;
    
    ncclose(nci);
end
