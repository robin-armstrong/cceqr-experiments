using LinearAlgebra

include("cceqr_utils.jl")

"""
cceqr!(A; k = minimum(size(A)), rho = 1e-4, full = false)

Compute the first `k` entries of the column permutation for a CPQR factorization
of `A`, modifying `A` in place. Use a "collect, commit, expand" strategy with block
proportion `rho`. If `full == true`, then apply Householder reflections to all
columns yielding a complete `R` factor. Returns the column permutation, the number of
cycles used, the average pivoting block size per cycle, and the final size of the tracked set.
"""
function cceqr!(A::Matrix{Float64};
                k::Int64 = minimum(size(A)),
                rho::Float64 = 1e-4,
                full::Bool = false)

    m, n = size(A)

    if k < 1
        throw(ArgumentError("k must be a positive integer"))
    elseif k > min(m, n)
        throw(ArgumentError("k cannot exceed the smaller dimension of the input matrix"))
    elseif rho*(1 - rho) <= 0
        throw(ArgumentError("rho must lie strictly between 0 and 1"))
    end

    gamma = zeros(n)         # residual column norms
    jpvt  = Array(1:n)       # column pivot vector
    s     = 0                # number of skeleton columns chosen so far
    t     = n                # number of currently tracked columns
    V     = zeros(m, k)      # storage for Householder reflectors
    T     = zeros(k, k)      # storage for T factor in compact WY representation

    avg_block = 0.

    # determine squared column norms

    broadcast!(norm, gamma, eachcol(A))
    gamma .*= gamma
    
    # compute column permutation in cycles

    cycle  = 0
    mu     = 0.

    while s < k
        cycle += 1

        # select a block from the non-skeleton tracked set to factorize
        b          = 1 + floor(Int64, rho*(t-1))
        avg_block += b
        delta      = order_reblock!(A, jpvt, s+1, s+t, gamma, b)

        (cycle == 1) && (t = b+1)

        # factorize candidate columns with GEQP3

        block   = A[(s+1):m, (s+1):(s+b)]
        min_dim = minimum(size(block))
        qrobj   = qr!(block, ColumnNorm())

        # decide how many pivot choices to commit. Note that mu = 0 at
        # the first cycle, but this is not a problem, for at the first
        # cycle delta serves as an upper bound on *all* non-candidate
        # residual norms

        c = sum(diag(qrobj.factors).^2 .>= max(delta, mu))
        c = min(c, k-s)

        # permute columns accordingly

        swap_cols!(A, jpvt, gamma, s+1, qrobj.p)

        # update the orthogonal factor

        update_q!(T, V, qrobj.factors, qrobj.Q.τ, s+1, s+c)

        # apply new reflectors to the tracked set
        
        apply_qt!(A, V, T, s+1, s+c, s+1, s+t)

        # check to see if we've chosen enough skeleton columns

        (s+c == k) && break

        # determine a lower bound on the maximum column norm orthogonal
        # to the range of A[:, 1:(s+c)], starting by measuring residual
        # norms in the block we just factorized.

        maxres = 0.
        
        for j = (s+c+1):(s+t)
            col       = view(A, (s+1):(s+c), j)
            gamma[j] -= norm(col)^2
            maxres    = max(maxres, gamma[j])
        end

        s += c
        t -= c
        
        if s+t < n
            # decide which remaining columns to bring into the tracked set

            r, mu = threshold_reblock!(A, jpvt, s+t+1, n, gamma, maxres)

            # measure their residual norms

            (r > 0) && apply_qt!(A, V, T, 1, s, s+t+1, s+t+r)

            for j = (s+t+1):(s+t+r)
                col       = view(A, 1:s, j)
                gamma[j] -= norm(col)^2
            end

            t  += r
        end
    end

    if full
        apply_qt!(A, V, T, 1, s, s+t+1, n)
    end

    avg_block /= cycle

    return jpvt[1:k], cycle, avg_block, s+t
end
