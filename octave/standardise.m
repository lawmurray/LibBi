% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1687 $
% $Date: 2011-06-28 11:46:45 +0800 (Tue, 28 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} standardise (@var{X}, @var{mu}, @var{Sigma})
%
% Standardise rows of @var{X} using given mean and standard deviation
% vectors.
%
% @itemize
% @bullet{ @var{X}}
%
% @bullet{ @var{mu} Mean vector.}
%
% @bullet{ @var{Sigma} Covariance matrix.}
% @end deftypefn
%
function Z = standardise (X, mu, Sigma)
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
    if columns (X) > 0 
        if columns (X) != columns (mu)
            error ('mu should have same number of columns as Z');
        end
        if columns (X) != columns (Sigma)
            error ('Sigma should have same number of columns as Z');
        end
        
        Mu = repmat(mu, rows(X), 1);
        invU = chol(cholinv(Sigma));
        Z = (X - Mu)*invU;
    else
        Z = [];
    end
end
