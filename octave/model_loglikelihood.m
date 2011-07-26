% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: -1 $
% $Date: $

% -*- texinfo -*-
% @deftypefn {Function File} model_loglikelihood (@var{in}, @var{invars}, @var{coords}, @var{M})
%
% Construct model for spatial exploration of likelihood.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% likelihood.}
%
% @bullet{ @var{invars} Name of variables from input file to include.
%
% @bullet{ @var{coords} Cell array giving vectors of spatial
% coordinates of zero to three elements, giving the x, y and z coordinates of
% the corresponding variable in @var{invars} to plot.}
%
% @bullet{ @var{M} Number of repeated likelihood computations for each
% sample in file.
% @end itemize
% @end deftypefn
%
function model = model_loglikelihood (in, invars, coords, M)
    % check arguments
    if nargin != 4
        print_usage ();
    end
        
    % read in log-likelihoods
    nc = netcdf(in, 'r');
    P = length(nc('np'));
    ll = nc{'loglikelihood'}(1:P);
    ll = reshape(ll, M, length(ll)/M);
    
    mx = max(ll);
    l = exp(ll - repmat(mx, M, 1));
    y = log(mean(l))' + mx';
    
    % (standardised) support points
    X = [];
    for i = 1:length(invars)
        if length(coords) >= i
            X = [ X, read_var(nc, invars{i}, coords{i}, [1:M:P], 1) ];
        else
            X = [ X, read_var(nc, invars{i}, [], [1:M:P], 1)(:) ];
        end
    end
    mu = mean(X);
    sigma = std(X);
    X = (X - repmat(mu, rows(X), 1))./repmat(sigma, rows(X), 1);
        
    % result structure
    model.mu = mu;
    model.sigma = sigma;
    model.X = X;
    model.y = y;
    
    % remove any NaNs and infs
    is = find(isfinite (model.y));
    js = find(sum (isfinite (model.X)));
    is = unique([is(:); js(:)]);
    
    model.X = model.X(is,:);
    model.y = model.y(is);
end
