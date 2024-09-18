using LinearAlgebra

### Computes a low-rank factorization K ~ X*X', where K is a kernel evaluation
### matrix given by K[i,j] = kfunc(norm(data[:, s[i]] - data[:, s[j]])), and kfunc
### defines a positive-definite kernel. The approximation factor X has dimension
### (n, k), where n = size(data, 2) and k is the approximation rank. This
### function uses a partial Cholesky factorization with adaptive random pivoting;
### see Chen, Epperly, Tropp, and Webber, 2023. All randomness is drawn from rng.

function rpchol!(rng, k, data, s, kfunc, X)
    n     = length(s)
    kdiag = kfunc(0.)*ones(n)

    for j = 1:k
        fprintln("        RPCHOL: filling Cholesky column "*string(j)*" of "*string(k))

        p     = sample(rng, 1:n, Weights(kdiag))
        pivot = data[:, s[p]]
        col   = view(X, :, j)
        
        for i = 1:n
            col[i] = kfunc(norm(pivot - data[:,s[i]]))
        end

        if j > 1
            V1    = view(X, :, 1:(j-1))
            V2    = view(X, p, 1:(j-1))
            mul!(col, V1, V2, -1., 1.)
        end

        col   ./= sqrt(col[p])
        kdiag .-= col.^2

        broadcast!(max, kdiag, kdiag, 0.)
    end
end
