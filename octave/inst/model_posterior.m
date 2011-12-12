% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: -1 $
% $Date: $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{model} = } model_posterior (@var{ins}, @var{invars}, @var{coords})
%
% Construct model for spatial exploration of posterior.
%
% @itemize
% @itemize @bullet{ @var{ins} Input file name, or cell array of file
% names. Gives the name of NetCDF files output by mcmc.}
%
% @bullet{ @var{invars} Name of variables from input file to include.
%
% @bullet{ @var{coords} Cell array giving vectors of spatial
% coordinates of zero to three elements, giving the x, y and z coordinates of
% the corresponding variable in @var{invars} to plot.}
% @end itemize
% @end deftypefn
%
function model = model_posterior (in, invars, coords, rang)
    % check arguments
    if nargin < 2 || nargin > 4
        print_usage ();
    end
    if nargin < 3
        coords = [];
    end
    if nargin < 4
        rang = [];
    end
    if iscell(ins) && !iscellstr(ins)
        error ('ins must be a string or cell array of strings');
    elseif ischar(ins)
        ins = { ins };
    end

    % read in support points
    Xs = [];
    for j = 1:length(ins)
        in = ins{j};
        nc = netcdf(in, 'r');
        P = length(nc('np'));
        rg = rang;
        if length (rg) == 0
            rg = [1:P];
        end

        X = [];
        for i = 1:length(invars)
            if length(coords) >= i
                X = [ X, read_var(nc, invars{i}, coords{i}, rg, 1) ];
            else
                X = [ X, read_var(nc, invars{i}, [], rg, 1)(:) ];
            end
        end
        Xs = [ Xs; X ];
    end
    
    % standardise support points
    mu = mean(Xs);
    Sigma = cov(Xs);
    Xs = standardise(Xs, mu, Sigma);
        
    % result structure
    model.type = 'posterior';
    model.mu = mu;
    model.Sigma = Sigma;
    model.X = Xs;
        
    % remove any NaNs and infs
    is = find(sum (isfinite (model.X), 2));
    model.X = model.X(is,:);
end
