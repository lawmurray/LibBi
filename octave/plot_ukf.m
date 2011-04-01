% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_ukf (@var{in}, @var{invars})
%
% Plot output of the ukf program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% ukf.}
%
% @bullet{ @var{invars} Cell array of strings naming the variables
% of this file to plot. Empty strings may be used to produce empty plots.}
% @end itemize
% @end deftypefn
%
function plot_ukf (in, invars)
    % check arguments
    if (nargin != 2)
        print_usage ();
    end

    % input file
    nci = netcdf(in, 'r');

    t = nci{'time'}(:)'; % times
    P = [0.025 0.5 0.975]; % quantiles (median and 95%)
    Q = zeros(length(t), length(P));
    for i = 1:length(invars)
        if (!strcmp(invars{i}, ''))
            id = nci{invars{i}}(:) + 1;
            mu = nci{'filter.mu'}(:,id);
            sigma = sqrt(nci{'filter.Sigma'}(:,id,id));
            for n = 1:length(t)
                Q(n,:) = logninv(P, mu(n), sigma(n));
            end
            
            % plot
            subplot(length(invars), 1, i);
            hold on;
            area_between(t, Q(:,1), Q(:,3), watercolour(6, 0.5));
            plot(t, Q(:,2), 'linewidth', 3, 'color', watercolour(6));
            title(invars{i});
            plot_defaults;
        end
    end
    
    ncclose(nci);
end
