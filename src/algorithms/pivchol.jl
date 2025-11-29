using LinearAlgebra

### Computes a low-rank factorization K ~ X*X', where K is a kernel evaluation
### matrix given by K[i,j] = kfunc(norm(data[:, samp[i]] - data[:, samp[j]])),
### and kfunc defines a positive-definite kernel. The approximation factor X has
### dimension (n, d), where n = size(data, 2) and d is the approximation rank. This
### function uses a partial Cholesky factorization pivoting, much in the spirit
### of Chen, Epperly, Tropp, and Webber, 2023, but deterministic.

function pivchol!(d, data, kfunc, X)
    n     = size(data, 1)
    kdiag = kfunc(0.)*ones(n)

    for j = 1:d
        fprintln("        RPCHOL: filling Cholesky column "*string(j)*" of "*string(d))

        p     = findmax(kdiag)[2]
        pivot = data[p,:]
        col   = zeros(n)
        
        for i = 1:n
            col[i] = kfunc(norm(pivot - data[i,:]))
        end

        if j > 1
            col[:] = col - X[:, 1:(j-1)]*X[p, 1:(j-1)]
        end
        
        col[:]   = col/sqrt(col[p])
        X[:,j]   = col
        kdiag[:] = kdiag - col.^2
        
        broadcast!(max, kdiag, kdiag, 0.)
    end
end
