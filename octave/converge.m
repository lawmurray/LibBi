% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1687 $
% $Date: 2011-06-28 13:46:45 +1000 (Tue, 28 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} converge (@var{in}, @var{invars},  @var{rang})
%
% Plot progress of the $\hat{R}^p$ statistic of Brooks & Gelman (1998).
%
% @itemize
% @bullet{ @var{ins} Cell array giving names of input files, each the name of
% a NetCDF file output by mcmc.}
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
function Rp = converge (ins, invars, coords, rang)    
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
    nci = netcdf(ins{1}, 'r');

    % data
    C = length(ins);
    N = length(invars);
    P = length (nci('np'));
    if length (rang) == 0
        rang = [1:P];
    end
    ncclose(nci);

    X = cell(C, 1);
    for k = 1:C
        nci = netcdf(ins{k}, 'r');
        X{k} = zeros(N, P);
        for i = 1:N
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
                    X{k}(i,:) = x;
                end
            end
        end
        ncclose(nci);
    end
        
    % means and covariances, indexed by chain then iteration
    mu = zeros(N, P, C);
    Sigma = zeros(N, N, P, C);

    for k = 1:C
        seq = [ 1:P ];
        halfseq = ceil(seq / 2);
        
        % cumulatives
        cum_mu = zeros(N, P);
        cum_Sigma = zeros(N, N, P);

        cum_mu = cumsum(X{k}, 2);
        for p = 1:P
            x = X{k}(:,p);
            cum_Sigma(:,:,p) = x*x';
        end
        cum_Sigma = cumsum(cum_Sigma, 3);
        
        % means and covariances
        mu1 = (cum_mu(:,seq) - cum_mu(:,halfseq)) ./ ...
            repmat(seq - halfseq, N, 1);
        Sigma1 = (cum_Sigma(:,:,seq) - cum_Sigma(:,:,halfseq));
        
        for p = 1:P
            x = mu1(:,p);
            n = seq(p) - halfseq(p);
            if seq(p) - halfseq(p) > 1
                Sigma1(:,:,p) /= n - 1;
                Sigma1(:,:,p) -= n/(n - 1)*x*x';
            else
                Sigma(:,:,p) = 0;
            end
        end
            
        mu(:,:,k) = mu1;
        Sigma(:,:,:,k) = Sigma1;
    end

    % scalar comparison
    W = squeeze(mean(Sigma, 4));
    Rp = zeros(P,1);
    for p = 1:P
        [Wp, s] = chol(squeeze(W(:,:,p)));
        if s == 0 % has Cholesky factorisation
            invWp = chol2inv(Wp);
            Bonp = cov(squeeze(mu(:,p,:))');
            lambda1 = max(eig(invWp*Bonp));
            Rp(p) = (P - 1)/P + (C + 1)/C*lambda1;
        end
    end
end
