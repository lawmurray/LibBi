% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} hist_ukf (@var{in}, @var{invars}, @var{m}, @var{n})
%
% Plot histogram of parameter samples output by ukf program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% ukf.}
%
% @bullet{ @var{invars} Cell array of strings naming the variables
% of this file to plot. Empty strings may be used to produce empty plots.}
%
% @bullet{ @var{m} Number of rows in plot.}
%
% @bullet{ @var{n} Number of columns in plot.}
% @end itemize
% @end deftypefn
%
function hist_ukf (in, invars, m, n, logn)
    % constants
    THRESHOLD = 5e-3; % threshold for bin removal start and end
    BINS = 20;

    % check arguments
    if (nargin < 2 || nargin > 5)
        print_usage ();
    elseif (nargin < 3)
        m = ceil(sqrt(length(invars)));
        n = m;
    elseif (nargin < 4)
        n = m;
    end
    if nargin < 5
        log = 0;
    end

    % input file
    nci = netcdf(in, 'r');

    for i = 1:length(invars)
        if (!strcmp(invars{i}, ''))
            % read
            id = nci{invars{i}}(end) + 1;
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
            subplot(m, n, i);
            hold on;
            plot(x, y, 'linewidth', 3, 'color', watercolour(3));
            grid on;
            title(invars{i});
        end
    end
    
    ncclose(nci);
end
