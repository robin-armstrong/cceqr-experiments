using LinearAlgebra

include("tcpqr_utils.jl")

"""
tcpqr!(A; k = minimum(size(A)), mu = .9)

Compute the first `k` entries of the column permutation for a CPQR factorization
of `A`, modifying `A` in place. Use thresholded updates with relative threshold `mu`.
Returns the column permutation, the number of cycles used, the average pivoting block
size per cycle, and the final size of the active set. See also `tcpqr`.
"""
function tcpqr!(A::Matrix{Float64}; k::Int64 = minimum(size(A)), mu::Float64 = .9)
    m, n = size(A)

    if k < 1 
        throw(ArgumentError("k must be a positive integer"))
    elseif k > min(m, n)
        throw(ArgumentError("k cannot exceed the smaller dimension of the input matrix"))
    elseif mu*(1 - mu) <= 0
        throw(ArgumentError("mu must lie strictly between 0 and 1"))
    end

    gamma = zeros(n)         # residual column norms
    lower = 0.               # lower bound on maximum residual column norm
    jpvt  = Array(1:n)       # column pivot vector
    skel  = 0                # number of skeleton columns chosen so far
    act   = n                # number of currently active columns
    V     = zeros(m, k)      # storage for Householder reflectors
    T     = zeros(k, k)      # storage for T factor in compact WY representation

    avg_block = 0.

    # determine column norms

    for j = 1:n
        gamma[j] = norm(A[:,j])^2
        
        if gamma[j] >= lower
            lower = gamma[j]
        end
    end
    
    # compute column permutation in cycles

    cycle = 0

    while skel < k
        cycle += 1
        delta  = mu*lower

        # select a block from the active set to factorize

        b          = reblock!(A, jpvt, skel+1, act, gamma, delta)
        avg_block += b
        (cycle == 1) && (act = b)

        # factorize candidate columns with DGEQP3

        block   = A[(skel+1):m, (skel+1):(skel+b)]
        min_dim = minimum(size(block))
        qrobj   = qr!(block, ColumnNorm())

        # figure out how many pivot choices are usable

        t = sum(diag(qrobj.factors).^2 .>= delta)
        t = min(t, k-skel)

        # permute columns accordingly

        swap_cols!(A, jpvt, gamma, skel+1, qrobj.p)

        # check to see if we've chosen enough skeleton columns

        (skel+t == k) && break

        # update the orthogonal factor

        V[(skel+1):m, (skel+1):(skel+t)] = qrobj.factors[:, 1:t]
        
        for i = (skel+1):(skel+t)
            V[i, i] = 1.
            fill!(view(V, 1:(i-1), i), 0.)
        end

        tau = qrobj.Q.τ[1:t]
        fill_t!(T, V, tau, skel+1, skel+t)

        # apply new reflectors to the active set
        
        apply_qt!(A, V, T, skel+1, skel+t, skel+1, act)

        # determine a lower bound on the maximum column norm orthogonal
        # to the range of A[:, 1:(skel+t)]

        if skel+t == act
            # in this case, the entire active set is now in the skeleton,
            # and we must update the norm of a non-active column 

            j     = skel+t+1                      # first non-active index
            col   = reshape(A[:, j], (m, 1))
            lower = norm(col[(skel+t+1):m])^2
        else
            lower = 0.
            
            for j = (skel+t+1):act
                col      = view(A, (skel+t+1):m, j)
                gamma[j] = norm(col)^2
                lower    = max(lower, gamma[j])
            end
        end

        # bring new columns into the active set and apply reflectors.

        act_old = act
        act    += reblock!(A, jpvt, act_old+1, n, gamma, mu*lower)

        if act > act_old
            apply_qt!(A, V, T, 1, skel+t, act_old+1, act)
        end

        # update residual column norms
        for j = (act_old+1):act
            col       = view(A, 1:(skel+t), j)
            gamma[j] -= norm(col)^2
            lower     = max(lower, gamma[j])
        end

        skel += t
    end

    avg_block /= cycle

    return jpvt[1:k], cycle, avg_block, act
end

"""
tcpqr(A; k = minimum(size(A)), mu = .9)

Compute the first `k` entries of the column permutation for a CPQR factorization of `A`,
using thresholded updates with relative threshold `mu`. Returns the column permutation,
the number of cycles used, the average pivoting block size per cycle, and the final size
of the active set. See also `tcpqr!`.
"""
function tcpqr(A::Matrix{Float64}; k::Int64 = minimum(size(A)), mu::Float64 = .9)
    return tcpqr!(deepcopy(A); k = k, mu = mu)
end
