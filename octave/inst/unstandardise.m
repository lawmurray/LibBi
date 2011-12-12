% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1687 $
% $Date: 2011-06-28 11:46:45 +0800 (Tue, 28 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} unstandardise (@var{Z}, @var{mu}, @var{Sigma})
%
% Unstandardise rows of @var{Z} using given mean and standard deviation
% vectors.
%
% @itemize
% @bullet{ @var{Z}}
%
% @bullet{ @var{mu} Mean vector.}
%
% @bullet{ @var{Sigma} Covariance matrix.}
% @end deftypefn
%
function X = unstandardise (Z, mu, Sigma)
    % check arguments
    if nargin != 3
        print_usage ();
    end
    if !isvector (mu)
        error ('mu should be row vector');
    else
        mu = mu(:)'; % ensure row vector
    end
    if !ismatrix (Sigma)
        error ('Sigma should be a matrix');
    end
    if columns (Z) > 0 
        if columns (Z) != columns (mu)
            error ('mu should have same number of columns as Z');
        end
        if columns (Z) != columns (Sigma)
            error ('Sigma should have same number of columns as Z');
        end
        
        Mu = repmat(mu, rows(Z), 1);
        U = chol(Sigma);
        X = Z*U + Mu;
    else
        X = [];
    end
end
