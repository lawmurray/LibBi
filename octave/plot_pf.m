% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1390 $
% $Date: 2011-04-11 16:06:40 +0800 (Mon, 11 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} plot_pf (@var{in}, @var{invars}, @var{coord})
%
% Plot output of the pf program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% pf.}
%
% @bullet{ @var{invar} Name of variable from input file to plot.
%
% @bullet{ @var{coord} (optional) Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{invar} to plot.}
% @end itemize
% @end deftypefn
%
function plot_pf (in, invar, coord)
    % check arguments
    if nargin < 2 || nargin > 3
        print_usage ();
    end
    if nargin < 3
        coord = [];
    elseif !isvector (coord) || length (coord) > 3
        error ('coord should be a vector with at most three elements');
    end
    
    % input file
    nci = netcdf(in, 'r');

    % data
    t = nci{'time'}(:)'; % times
    q = [0.025 0.5 0.975]'; % quantiles (median and 95%)
    P = nci('np')(:);
    T = nci('nr')(:);

    X = read_var (nci, invar, [1:P], coord);
    Ws = exp(nci{'logweight'}(:,:));
    Q = zeros (rows (X), length(q));
    
    [X I] = sort (X, 2);
    Ws = Ws(I);
    Wt = sum(Ws, 2);
    Wc = cumsum(Ws, 2) ./ repmat(Wt, 1, P);
   
    for i = 1:T
        % build weighted empirical cdf
        for j = 1:length(q)
            is = find(Wc(i,:) < q(j));
            if length(is) > 0
                k = is(end) + 1;
            else
                k = 1;
            end
            Q(i,j) = X(i,k);
        end
    end
    
    % plot
    ish = ishold;
    area_between(t, Q(:,1), Q(:,3), watercolour(2, 0.5));
    hold on
    plot(t, Q(:,2), 'linewidth', 3, 'color', watercolour(2));
    if ish
        hold on
    else
        hold off
    end
    %title(nice_name(name, dims));
    %plot_defaults;
    
    ncclose(nci);
end
