function m = mingp (z, model)
    mu = model.meanfunc(model.hyp.mean, z);
    c = model.covfunc(model.hyp.cov, model.X, z);
    m = c'*model.k + mu;
end
