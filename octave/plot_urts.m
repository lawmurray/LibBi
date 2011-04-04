% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_urts (@var{in}, @var{invars})
%
% Plot output of the urts program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% urts.}
%
% @bullet{ @var{invars} Cell array of strings naming the variables
% of this file to plot. Empty strings may be used to produce empty plots.}
% @end itemize
% @end deftypefn
%
function plot_urts (in, invars)
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
            mu = nci{'smooth.mu'}(:,id);
            sigma = sqrt(nci{'smooth.Sigma'}(:,id,id));
            for n = 1:length(t)
                Q(n,:) = logninv(P, mu(n), sigma(n));
            end
            
            % plot
            if length(invars) > 1
                subplot(length(invars), 1, i);
            end
            hold on;
            grid on;
            area_between(t, Q(:,1), Q(:,3), watercolour(6, 0.5));
            plot(t, Q(:,2), 'linewidth', 3, 'color', watercolour(6));
            title(invars{i});
            plot_defaults;
        end
    end
    
    ncclose(nci);
end
