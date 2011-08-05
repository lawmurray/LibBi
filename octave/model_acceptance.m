% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} {@var{model} = } model_acceptance (@var{ins}, @var{invars}, @var{coords}, @var{M})
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
% @end itemize
% @end deftypefn
%
function model = model_acceptance (ins, invars, coords, M)
    % check arguments
    if nargin != 4
        print_usage ();
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
        ll = reshape(ll, M, length(ll)/M);
    
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
        Xs = [ Xs; X ];
        alphas = [ alphas; alpha ];
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
