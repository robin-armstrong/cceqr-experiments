using LinearAlgebra

### Computes a low-rank factorization K ~ X*X', where K is a kernel evaluation
### matrix given by K[i,j] = kfunc(norm(data[:, samp[i]] - data[:, samp[j]])),
### and kfunc defines a positive-definite kernel. The approximation factor X has
### dimension (n, d), where n = size(data, 2) and d is the approximation rank. This
### function uses a partial Cholesky factorization with adaptive random pivoting;
### see Chen, Epperly, Tropp, and Webber, 2023. All randomness is drawn from rng.

function rpchol!(rng, d, data, samp, kfunc, X)
    n     = length(samp)
    kdiag = kfunc(0.)*ones(n)

    for j = 1:d
        fprintln("        RPCHOL: filling Cholesky column "*string(j)*" of "*string(d))

        p     = sample(rng, 1:n, Weights(kdiag))
        pivot = data[:, samp[p]]
        col   = view(X, :, j)
        
        for i = 1:n
            col[i] = kfunc(norm(pivot - data[:, samp[i]]))
        end

        if j > 1
            V1 = view(X, :, 1:(j-1))
            V2 = view(X, p, 1:(j-1))
            mul!(col, V1, V2, -1., 1.)
        end

        col   ./= sqrt(col[p])
        kdiag .-= col.^2

        broadcast!(max, kdiag, kdiag, 0.)
    end
end
