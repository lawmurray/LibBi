% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} hist_mcmc (@var{in}, @var{invars}, @var{m}, @var{n})
%
% Plot histogram of parameter and initial condition samples output by mcmc
% program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% mcmc.}
%
% @bullet{ @var{invars} Cell array of strings naming the variables
% of this file to plot. Empty strings may be used to produce empty plots.}
%
% @bullet{ @var{m} Number of rows in plot.}
%
% @bullet{ @var{n} Number of columns in plot.}
%
% @bullet{ @var{ref} Reference file. Gives the name of a NetCDF file output
% by mcmc to use as a reference for positioning bins (TODO).
% @end itemize
% @end deftypefn
%
function hist_mcmc (in, invars, m, n)
    % constants
    THRESHOLD = 5e-3; % threshold for bin removal start and end
    BINS = 20;

    % check arguments
    if (nargin < 2 || nargin > 4)
        print_usage ();
    elseif (nargin < 3)
        m = ceil(sqrt(length(invars)));
        n = m;
    elseif (nargin < 4)
        n = m;
    end

    % input file
    nci = netcdf(in, 'r');
    
    for i = 1:length(invars)
        if (!strcmp(invars{i}, ''))
            % read samples
            numdims = ndims(nci{invars{i}}); % always >= 2
            if (numdims == 2 && columns(nci{invars{i}}) == 1)
                % parameter, take all values
                data = nci{invars{i}}(:);
            elseif numdims == 2
                % state variable, take initial value
                data = nci{invars{i}}(1,:);
            else
                error (sprintf(['%s is incompatible, must have 1 or 2 ' ...
                                'dimensions'], invars{i}));
            end
        
            % bin
            [nn,xx] = hist(data, BINS);

            % clean up outlier bins from either end
            mask = nn > THRESHOLD*length(data);
            found = find(mask, length(mask), "first");
            if length(found) > 1
                % have outliers
                first = found(1);
                last = find(mask, length(mask), "last")(end);
                if last == first
                    last = last + 1;
                end
                mask(first:last) = 1; % only want to remove from start and end
                xx = xx(mask);
                nn = nn(mask);
            end


            % recompute reference range to restore full number of bins
            [nn,xx] = hist(data, xx(1):(xx(end)-xx(1))/(BINS-1):xx(end));
            [mm,yy] = hist(data, xx);
            % ^ first above should be with reference range
            
            % scale
            xsize = max(xx) - min(xx); % range of x values in histogram
            xdelta = xsize / 100; % res for prior distro plot
            ysize = length(data) / BINS; % average bar height

            % prior
            %x = min([ xx, truth(i) ]):xdelta:max([ xx, truth(i) ]);
            %if i == 3
            %    y = normpdf(x, mu0(i), sigma0(i));
            %else
            %    y = lognpdf(x, mu0(i), sigma0(i));
            %end
            %peak = max(max(nn) / sum(xsize/BINS*mm), max(y));
                        
            % plot
            if length(invars) > 1
                subplot(m, n, i);
            end
            hold on;
            h = bar(xx,mm / sum(xsize/BINS*mm), 1.0); % normalised histogram
            set(h, 'FaceColor', watercolour(6,0.5), ...
                   'EdgeColor', watercolour(6));
            title(invars{i});
            hist_defaults;
        end
    end
    
    ncclose(nci);
end
