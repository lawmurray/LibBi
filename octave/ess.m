% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1390 $
% $Date: 2011-04-11 16:06:40 +0800 (Mon, 11 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} ess (@var{lws})
%
% Compute effective sample size (ESS) for given log-weights.
%
% @itemize
% @bullet{ @var{lws} Log-weights.}
% @end deftypefn
%
function ss = ess (lws)
% check arguments
  if (nargin != 1)
      print_usage ();
  end
  
  mx = max(lws');
  Mx = repmat(mx', 1, columns(lws));
  ws = exp(lws - Mx);
  num = sum(ws, 2).^2;
  den = sum(ws.^2, 2);
  ss = num ./ den;
end
