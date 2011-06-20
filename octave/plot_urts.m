% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_urts (@var{in}, @var{invar}, @var{coord}, @var{islog})
%
% Plot output of the urts program.
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
%
% @bullet{ @var{islog} (optional) True if this is a log-variable, false
% otherwise.
% @end itemize
% @end deftypefn
%
function plot_urts (in, invar, coord, islog)
    % check arguments
    if nargin < 2 || nargin > 4
        print_usage ();
    end
    if nargin < 3
        coord = [];
    elseif !check_coord (coord)
        error ('coord should be a vector with at most three elements');
    end
    if nargin < 4
        islog = 0;
    end

    % input file
    nci = netcdf(in, 'r');
    if ncdimexists (nci, 'nx')
        xlen = length (nci('nx'));
    else
        xlen = 1;
    end
    if ncdimexists (nci, 'ny')
        ylen = length (nci('ny'));
    else
        ylen = 1;
    end
    if ncdimexists (nci, 'ny')
        ylen = length (nci('ny'));
    else
        ylen = 1;
    end
    
    if length(coord) == 3
        offset = coord(1) + coord(2)*xlen + coord(3)*xlen*ylen;
    elseif length(coord) == 2
        offset = coord(1) + coord(2)*xlen;
    elseif length(coord) == 1
        offset = coord(1);
    else
        offset = 1;
    end
    
    t = nci{'time'}(:)'; % times
    P = [0.025 0.5 0.975]; % quantiles (median and 95%)
    Q = zeros(length(t), length(P));

    id = nci{invar}(:) + offset;
    mu = nci{'smooth.mu'}(:,id);
    sigma = sqrt(nci{'smooth.Sigma'}(:,id,id));
    for n = 1:length(t)
        if islog
            if sigma(n) > 0
                Q(n,:) = logninv(P, mu(n), sigma(n));
            else
                Q(n,:) = exp(mu(n));
            end
        else
            if sigma(n) > 0
                Q(n,:) = norminv(P, mu(n), sigma(n));
            else
                Q(n,:) = mu(n);
            end
        end
    end
            
    % plot
    ish = ishold;
    if !ish
        clf % patch doesn't clear otherwise
    end
    area_between(t, Q(:,1), Q(:,3), watercolour(2));
    hold on;
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
