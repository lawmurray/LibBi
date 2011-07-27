% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: -1 $
% $Date: $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{model} = } model_posterior (@var{in}, @var{invars}, @var{coords})
%
% Construct model for spatial exploration of posterior.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% mcmc.}
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
        
    % read in posterior samples
    nc = netcdf(in, 'r');
    P = length(nc('np'));
    if length (rang) == 0
        rang = [1:P];
    end

    % (standardised) support points
    X = [];
    for i = 1:length(invars)
        if length(coords) >= i
            X = [ X, read_var(nc, invars{i}, coords{i}, rang, 1) ];
        else
            X = [ X, read_var(nc, invars{i}, [], rang, 1)(:) ];
        end
    end
    mu = mean(X);
    sigma = std(X);
    X = (X - repmat(mu, rows(X), 1))./repmat(sigma, rows(X), 1);
        
    % result structure
    model.type = 'posterior';
    model.mu = mu;
    model.sigma = sigma;
    model.X = X;
        
    % remove any NaNs and infs
    is = find(sum (isfinite (model.X), 2));
    model.X = model.X(is,:);
end
