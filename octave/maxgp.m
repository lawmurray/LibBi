function m = maxgp (z, hyp, inffunc, meanfunc, covfunc, likfunc, X, y)
    m = -gp (hyp, inffunc, meanfunc, covfunc, likfunc, X, y, z);
end
