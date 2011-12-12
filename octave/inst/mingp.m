function m = mingp (z, model)
    mu = feval(model.meanfunc, model.hyp.mean, z);
    c = feval(model.covfunc, model.hyp.cov, model.X, z);
    m = c'*model.k + mu;
end
