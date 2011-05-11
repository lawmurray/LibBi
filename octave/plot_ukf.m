% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_ukf (@var{in}, @var{invar}, @var{coord})
%
% Plot output of the ukf program.
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
function plot_ukf (in, invar, coord)
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
    xlen = nci('nx');
    ylen = nci('ny');
    zlen = nci('nz');
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
    mu = nci{'filter.mu'}(:,id);
    sigma = sqrt(nci{'filter.Sigma'}(:,id,id));
    for n = 1:length(t)
        if isreal(sigma(n))
            Q(n,:) = norminv(P, mu(n), sigma(n));
        else
            Q(n,:) = mu(n);
        end
    end
            
    % plot
    ish = ishold;
    if !ish
        clf % patch doesn't clear otherwise
    end
    area_between(t, Q(:,1), Q(:,3), watercolour(2, 0.5));
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
