using LinearAlgebra
using StatsBase
using Random

### Fills the (1:q, p:q) block of T for compact WY form using the Schreiber-van Loan algorithm.

function fill_t!(T::AbstractMatrix{Float64}, V::Matrix{Float64}, tau::Vector{Float64}, p::Int64, q::Int64)
    # filling the (p:q, p:q) block of T
    for s = 1:(q-p+1)
        i       = p+s-1
        T[i, i] = tau[s]
        
        if i > p
            c      = view(T, p:(i-1), i)
            V_prev = view(V, :, p:(i-1))
            V_new  = view(V, :, i)
            T_prev = UpperTriangular(view(T, p:(i-1), p:(i-1)))

            mul!(c, V_prev', V_new, -tau[s], 0.)
            lmul!(T_prev, c)
        end
    end
    
    # filling the (1:(p-1), p:q) block of T
    if p > 1
        T11 = UpperTriangular(view(T, 1:(p-1), 1:(p-1)))
        T22 = UpperTriangular(view(T, p:q, p:q))
        T12 = view(T, 1:(p-1), p:q)
        V1  = view(V, :, 1:(p-1))
        V2  = view(V, :, p:q)

        mul!(T12, V1', V2, -1., 0.)
        lmul!(T11, T12)
        rmul!(T12, T22)
    end
end

### Applies Qt to the (p:q, r:s) block of A, where Qt is defined by the p-th through q-th Householder reflectors
### in the compact WY form given by V and T.

function apply_qt!(A::Matrix{Float64}, V::Matrix{Float64}, T::Matrix{Float64}, p::Int64, q::Int64, r::Int64, s::Int64)
    m, n  = size(A)
    A1    = view(A, p:q, r:s)
    A2    = view(A, (q+1):m, r:s)
    T_sub = view(T, p:q, p:q)
    V1    = view(V, p:q, p:q)
    V2    = view(V, (q+1):m, p:q)
    W     = Matrix{Float64}(A[p:q, r:s]')

    rmul!(W, LowerTriangular(V1))
    mul!(W, A2', V2, 1., 1.)
    rmul!(W, UpperTriangular(T_sub))
    mul!(A1, V1, W', -1., 1.)
    mul!(A2, V2, W', -1., 1.)
end

### Loops through the (:, j_start:j_end) block of A and moves all columns to the front that have squared norm exceeding threshold
### delta. Modifies jpvt and gamma accordingly. Returns the number of columns that passed the threshold.

function threshold_reblock!(A::Matrix{Float64}, jpvt::Vector{Int64}, j_start::Int64, j_end::Int64, gamma::Vector{Float64}, delta::Float64)
    m, n   = size(A)
    tmpcol = zeros(m)
    blk    = 0

    for j = j_start:j_end
        if gamma[j] > delta
            blk += 1
            p    = j_start+blk-1
            
            tmp     = jpvt[p]
            jpvt[p] = jpvt[j]
            jpvt[j] = tmp

            tmp      = gamma[p]
            gamma[p] = gamma[j]
            gamma[j] = tmp

            tmpcol[:] = A[:,p]
            A[:,p]    = A[:,j]
            A[:,j]    = tmpcol
        end
    end

    return blk
end

### Loops through the (:, j0:end) block of A and moves r columns to the front, corresponding to the columns
### with the highest value or gamma. Modifies jpvt and gamma accordingly.

function order_reblock!(A::Matrix{Float64}, jpvt::Vector{Int64}, gamma::Vector{Float64}, j0::Int64, r::Int64)
    m, n   = size(A)
    tmpcol = zeros(m)
    samp   = partialsortperm(view(gamma, j0:n), 1:r, rev = true)

    for i = 1:r
        p = j0+samp[i]-1
        j = j0+i-1

        tmp     = jpvt[p]
        jpvt[p] = jpvt[j]
        jpvt[j] = tmp

        tmp      = gamma[p]
        gamma[p] = gamma[j]
        gamma[j] = tmp

        tmpcol[:] = A[:,p]
        A[:,p]    = A[:,j]
        A[:,j]    = tmpcol
    end
end

### Applies column permutation "perm" to the (:, j0:(j0+length(perm)-1)) block of A, modifying jpvt and gamma accordingly.

function swap_cols!(A::Matrix{Float64}, jpvt::Vector{Int64}, gamma::Vector{Float64}, j0::Int64, perm::Vector{Int64})
    n = size(A, 2)
    b = length(perm)

    permute!(view(jpvt, j0:(j0+b-1)), perm)
    permute!(view(gamma, j0:(j0+b-1)), perm)
    Base.permutecols!!(view(A, :, j0:(j0+b-1)), perm)
end
