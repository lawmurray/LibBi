% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} image_corr (@var{in}, @var{invars},  @var{rang})
%
% Plot correlation matrix of samples output by mcmc program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% mcmc.}
%
% @bullet{ @var{invars} Cell array giving names of variables to plot.
%
% @bullet{ @var{coords} Cell array giving coordinates of variables to plot,
% each element matching an element of @var{invars}.
%
% @bullet{ @var{rang} (optional) Vector of indices of samples to
% include. Useful for excluding burn-in periods, for instance.
% @end itemize
% @end deftypefn
%
function image_cor (in, invars, coords, rang)    
    % check arguments
    if nargin < 2 || nargin > 4
        print_usage ();
    end
    if nargin < 3
        coords = {};
    end
    if nargin < 4
        rang = [];
    end
    if !(length (coords) == 0 || length(coords) == length(invars))
        error ('Length of invars and coords must match');
    end
    
    % input file
    nci = netcdf(in, 'r');

    % data
    P = length (nci('np'));
    if length (rang) == 0
        rang = [1:P];
    end

    X = [];
    names = {};
    for i = 1:length(invars)
        invar = invars{i};
        
        if length (coords) > 0
            coords1 = coords{i};
            if rows (coords1) == 0
                coords1 = zeros(1, 0);
            end
        else
            coords1 = zeros(1, 0);
        end
        
        for j = 1:max(1, rows(coords1))
            coord = coords1(j,:);
            if !check_coord (coord)
                error ('Invalid coordinate');
            else
                x = read_var(nci, invar, coord, rang, 1);
                X = [ X, x ];
            end
            name = nice_name (invar, coord);
            names = { names{:}, name };
        end
    end
    ncclose(nci);        
        
    % compute correlation
    Cor = cor(X,X);
    
    % plot
    imagesc(Cor);
    colorbar;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', 1:length(names));
    set(gca, 'xticklabel', names);
    set(gca, 'ytick', 1:length(names));
    set(gca, 'yticklabel', names);
end
