% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1687 $
% $Date: 2011-06-28 11:46:45 +0800 (Tue, 28 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} unstandardise (@var{Z}, @var{mu}, @var{sigma})
%
% Unstandardise rows of @var{Z} using given mean and standard deviation
% vectors.
%
% @itemize
% @bullet{ @var{Z}}
%
% @bullet{ @var{mu} Mean vector.}
%
% @bullet{ @var{sigma} Standard deviation vector.}
% @end deftypefn
%
function X = unstandardise (Z, mu, sigma)
    % check arguments
    if nargin != 3
        print_usage ();
    end
    if !isvector (mu)
        error ('mu should be row vector');
    else
        mu = mu(:)'; % ensure row vector
    end
    if !isvector (sigma)
        error ('sigma should be vector');
    else
        sigma = sigma(:)';
    end
    if columns (Z) > 0 
        if columns (Z) != columns (mu)
            error ('mu should have same number of columns as Z');
        end
        if columns (Z) != columns (sigma)
            error ('sigma should have same number of columns as Z');
        end
        
        Mu = repmat(mu, rows(Z), 1);
        Sigma = repmat(sigma, rows(Z), 1);
        X = Z.*Sigma + Mu;
    else
        X = [];
    end
end
