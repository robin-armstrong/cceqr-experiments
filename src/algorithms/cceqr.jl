using LinearAlgebra

include("cceqr_utils.jl")

"""
cceqr!(A; k = minimum(size(A)), rho = 1e-4, eta = .1)

Compute the first `k` entries of the column permutation for a CPQR factorization
of `A`, modifying `A` in place. Use a "collect, commit, expand" strategy with block
proportion `rho` and expansion threshold `eta`. Returns the column permutation, the
number of cycles used, the average pivoting block size per cycle, and the final size
of the tracked set.
"""
function cceqr!(A::Matrix{Float64}; k::Int64 = minimum(size(A)), rho::Float64 = 1e-4, eta::Float64 = .1)
    m, n = size(A)

    if k < 1
        throw(ArgumentError("k must be a positive integer"))
    elseif k > min(m, n)
        throw(ArgumentError("k cannot exceed the smaller dimension of the input matrix"))
    elseif rho*(1 - rho) <= 0
        throw(ArgumentError("rho must lie strictly between 0 and 1"))
    elseif eta*(1 - eta) <= 0
        throw(ArgumentError("eta must lie strictly between 0 and 1"))
    end

    gamma = zeros(n)         # residual column norms
    jpvt  = Array(1:n)       # column pivot vector
    s     = 0                # number of skeleton columns chosen so far
    t     = n                # number of currently tracked columns
    V     = zeros(m, k)      # storage for Householder reflectors
    T     = zeros(k, k)      # storage for T factor in compact WY representation

    avg_block = 0.

    # determine column norms

    for j = 1:n
        gamma[j] = norm(A[:,j])^2
    end
    
    # compute column permutation in cycles

    cycle = 0

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

        # decide how many pivot choices to commit

        c = sum(diag(qrobj.factors).^2 .>= delta)
        c = min(c, k-s)

        # permute columns accordingly

        swap_cols!(A, jpvt, gamma, s+1, qrobj.p)

        # check to see if we've chosen enough skeleton columns

        (s+c == k) && break

        # update the orthogonal factor

        V[(s+1):m, (s+1):(s+c)] = qrobj.factors[:, 1:c]
        
        for i = (s+1):(s+c)
            V[i, i] = 1.
            fill!(view(V, 1:(i-1), i), 0.)
        end

        tau = qrobj.Q.τ[1:c]
        fill_t!(T, V, tau, s+1, s+c)

        # apply new reflectors to the tracked set
        
        apply_qt!(A, V, T, s+1, s+c, s+1, s+t)

        # determine a lower bound on the maximum column norm orthogonal
        # to the range of A[:, 1:(s+c)], starting by measuring residual
        # norms in the block we just factorized.

        lower = 0.
        
        for j = (s+c+1):(s+t)
            col      = view(A, (s+c+1):m, j)
            gamma[j] = norm(col)^2
            lower    = max(lower, gamma[j])
        end

        # if there are still non-tracked columns, then we choose a small 
        # number of them to make tracked, and we measure their residual
        # column norms as well to update the lower bound.

        if s+t < n
            r = ceil(Int64, rho*(n-s-c-t))
            order_reblock!(A, jpvt, s+t+1, n, gamma, r)
            apply_qt!(A, V, T, 1, s+c, s+t+1, s+t+r)

            for j = (s+t+1):(s+t+r)
                col      = view(A, (s+c+1):m, j)
                gamma[j] = norm(col)^2
                lower    = max(lower, gamma[j])
            end

            t += r

            # decide which remaining columns to bring into the tracked set

            r = threshold_reblock!(A, jpvt, s+t+1, n, gamma, (1-eta)*lower)

            # measure their residual norms

            (r > 0) && apply_qt!(A, V, T, 1, s+c, s+t+1, s+t+r)

            for j = (s+t+1):(s+t+r)
                col       = view(A, 1:(s+c), j)
                gamma[j] -= norm(col)^2
            end

            t  += r
        end

        s += c
        t -= c
    end

    avg_block /= cycle

    return jpvt[1:k], cycle, avg_block, s+t
end
