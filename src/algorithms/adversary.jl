using Hadamard

# Constructs an "adversarial" Hadamard-like test matrix of
# size (m, n). We require that m and n are both powers of 2,
# and that n > m.

function adversary(m, n)
    d   = div(n, m)
    tmp = zeros(m, n)
    A   = zeros(m, n)
    H   = hadamard(m)
    
    for i = 1:m
        idx_A      = ((i-1)*d + 1):(i*d)
        A[:,idx_A] = H[:, i]*ones(1, d)
    end
    
    D = Diagonal(reverse(1 .+ 1000*eps()*(1:n)))
    rmul!(A, D)

    return A
end
