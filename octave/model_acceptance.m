% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} {@var{model} = } model_acceptance (@var{ins}, @var{invars}, @var{coords}, @var{M}, @var{logs})
%
% Construct model for spatial exploration of acceptance rates.
%
% @itemize @bullet{ @var{ins} Input file name, or cell array of file
% names. Gives the name of NetCDF files output by likelihood.}
%
% @bullet{ @var{invars} Name of variables from input file to include.
%
% @bullet{ @var{coords} Cell array giving vectors of spatial
% coordinates of zero to three elements, giving the x, y and z coordinates of
% the corresponding variable in @var{invars} to plot.}
%
% @bullet{ @var{M} Number of repeated likelihood computations for each
% sample in file.
%
% @bullet{ @var{logs} Indices of variables for which to take logarithm
% before standardising.}
% @end itemize
% @end deftypefn
%
function model = model_acceptance (ins, invars, coords, M, logs)
    % check arguments
    if nargin < 4
        print_usage ();
    end
    if nargin < 5
        logs = [];
    end
    if iscell(ins) && !iscellstr(ins)
        error ('ins must be a string or cell array of strings');
    elseif ischar(ins)
        ins = { ins };
    end
        
    % read in log-likelihoods and support points
    Xs = [];
    alphas = [];
    for j = 1:length(ins)
        in = ins{j};
        nc = netcdf(in, 'r');
        P = length(nc('np'));
        ll = nc{'loglikelihood'}(1:P);
        ll = reshape(ll, M, P/M);
    
        % compute expected acceptance rates
        alpha = zeros(columns(ll), 1);
        for i = 1:length(alpha)
            l = ll(:,i);
            mx = max(l);
            lsm = mx + log(sum(exp(l - mx)));
            l = exp(l - lsm);
            l = sort(l, 'descend');
            b = cumsum(l);
            c = 1.0 - b;
            alpha(i) = (l'*b + sum(c))/M;
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
        Xs = [ Xs; X ];
        alphas = [ alphas; alpha ];
    end
    
    % log-variables
    for i = 1:length (logs)
        Xs(:,logs(i)) = log (Xs(:,logs(i)));
    end
    
    % standardise support points
    mu = mean(Xs);
    Sigma = cov(Xs);
    Xs = standardise(Xs, mu, Sigma);
        
    % result structure
    model.type = 'acceptance';
    model.mu = mu;
    model.Sigma = Sigma;
    model.X = Xs;
    model.y = log(alphas);
    
    % remove any NaNs and infs
    is = find(isfinite (model.y));
    js = find(sum (isfinite (model.X), 2));
    is = intersect(is(:), js(:));
    
    model.X = model.X(is,:);
    model.y = model.y(is);
end
