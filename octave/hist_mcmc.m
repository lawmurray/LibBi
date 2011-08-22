% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} hist_mcmc (@var{ins}, @var{invar}, @var{coord}, @var{ps}, @var{logn})
%
% Plot histogram of parameter samples output by mcmc program.
%
% @itemize
% @bullet{ @var{ins} Input files. Gives the names of NetCDF files output by
% mcmc.}
%
% @bullet{ @var{invar} Name of variable from input file to plot.
%
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
%
% @bullet{ @var{ps} (optional) Vector of indices of samples to
% plot. All samples plotted if not specified.
%
% @bullet{logn} (optional) True to histogram log of variable, false
% otherwise.
% @end itemize
% @end deftypefn
%
function hist_mcmc (ins, invar, coord, ps, logn)
    % constants
    THRESHOLD = 5e-3; % threshold for bin removal start and end
    BINS = 20;

    % check arguments
    if (nargin < 2 || nargin > 5)
        print_usage ();
    end
    if nargin < 3
        coord = [];
    end
    if nargin < 4
        ps = [];
    end
    if nargin < 5
        logn = 0;
    end
    if !check_coord (coord)
        error ('coord should be a vector with at most three elements');
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
        nci = netcdf (in, 'r');
        P = length (nci('np'));
        if length (ps) == 0
            ps = [1:P];
        end
    
        % read samples
        x = read_var (nci, invar, coord, ps, 1);
        X = [ X; x(:) ];
        
        ncclose (nci);
    end
    
    if logn
        X = log (X);
    end
    
    % bin
    [nn,xx] = hist(X, BINS);
    
    % clean up outlier bins from either end
    mask = nn > THRESHOLD*length(X);
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
    [nn,xx] = hist(X, xx(1):(xx(end)-xx(1))/(BINS-1):xx(end));
    [mm,yy] = hist(X, xx);
    % ^ first above should be with reference range
            
    % scale
    xsize = max(xx) - min(xx); % range of x values in histogram
    %xdelta = xsize / 100; % res for prior distro plot
    ysize = mean(mm); % average bar height

    % prior
    %x = min([ xx, truth(i) ]):xdelta:max([ xx, truth(i) ]);
    %if i == 3
    %    y = normpdf(x, mu0(i), sigma0(i));
    %else
    %    y = lognpdf(x, mu0(i), sigma0(i));
    %end
    %peak = max(max(nn) / sum(xsize/BINS*mm), max(y));
                        
    % plot
    h = bar(yy,mm/(xsize*ysize), 1.0); % normalised histogram
    set(h, 'FaceColor', fade(watercolour(2),0.5), 'EdgeColor', watercolour(2));
end
