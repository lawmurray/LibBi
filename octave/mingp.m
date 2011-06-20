function m = mingp (z, hyp, inffunc, meanfunc, covfunc, likfunc, X, y)
%H = convhulln ([ z; H ]);
%   if sum(H(:) == 1) == 0
       m = gp (hyp, inffunc, meanfunc, covfunc, likfunc, X, y, z);
%   else
%        m = 2*max(y(:)); % gp will never be greater than this over
%                         % constrained domain
%    end
end
