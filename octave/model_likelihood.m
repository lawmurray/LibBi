% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} model_likelihood (@var{in}, @var{invars}, @var{coords}, @var{M})
%
% Krig likelihood noise.
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
function model = model_likelihood (in, invars, coords, M)
    % read in log-likelihoods
    nc = netcdf(in, 'r');
    ll = nc{'loglikelihood'}(:);
    ll = reshape(ll, M, length(ll)/M);
    P = length(nc('np'));
    
    % compute expected acceptance rates
    alpha = zeros(columns(ll), 1);
    for i = 1:length(alpha)
        l = ll(:,i);
        T = repmat(l, 1, M) - repmat(l', M, 1);
        T = (T < 0).*T; % max zero
        T = exp(T)/M;
        T(1:M+1:end) = ones(M, 1) - (sum(T,1)' - diag(T));

        % simulate to equilibrium
        U1 = T;
        U2 = U1*U1;
        n = 0;
        while (n < 32 && norm(U1 - U2, 'fro') > 0)
            U1 = U2;
            U2 = U1*U1;
            U2 = U2./repmat(sum(U2,1), M, 1); % renormalise
            ++n;
        end
        l = U2*ones(M, 1)/M;
        alpha(i) = l'*(1.0 - diag(T) + 1/M);
    end
    
    % support points
    X = [];
    for i = 1:length(invars)
        if length(coords) >= i
            X = [ X, read_var(nc, invars{i}, coords{i}, [1:M:P], 1) ];
        else
            X = [ X, read_var(nc, invars{i}, [], [1:M:P], 1)(:) ];
        end
    end
    
    % result structure
    model.X = X;
    model.logalpha = log(alpha);
end
