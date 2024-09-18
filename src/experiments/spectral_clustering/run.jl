using LinearAlgebra
using CairoMakie
using StatsBase
using Random
using JLD2

include("../../algorithms/cceqr.jl")
include("../../algorithms/rpchol.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng = MersenneTwister(1)

n_centers = 5
n_shells  = 4
radius    = 10
noise     = .2
n_points  = round.(Int64, exp10.(range(4, 5, 15)))     # range of dataset sizes

kernel    = "inv-1"     # type of kernel function      
bandwidth = 1.          # bandwidth of kernel function

eta       = 1e-4        # controls threshold value for CCEQR
rho       = 1e-4        # controls selection of columns for Householder reflection
numtrials = 100         # algorithm trials per separation value

plot_only = false       # if "true" then data will be read from disk and not regenerated

destination = "src/experiments/spectral_clustering/test"
readme      = "Getting the script to work."

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

if !plot_only
    function fprintln(s)
        println(s)
        flush(stdout)
    end

    function run_cluster_experiment(rng, n_centers, n_shells, radius, noise,
                                    n_points, kernel, bandwidth, eta, rho, numtrials,
                                    plot_only, destination, readme)
        
        logstr  = "rng       = "*string(rng)*"\n"
        logstr *= "n_centers = "*string(n_centers)*"\n"
        logstr *= "n_shells  = "*string(n_shells)*"\n"
        logstr *= "radius    = "*string(radius)*"\n"
        logstr *= "noise     = "*string(noise)*"\n"
        logstr *= "n_points  = "*string(n_points)*"\n"
        logstr *= "kernel    = "*kernel*"\n"
        logstr *= "bandwidth = "*string(bandwidth)*"\n"
        logstr *= "eta       = "*string(eta)*"\n"
        logstr *= "rho       = "*string(rho)*"\n"
        logstr *= "numtrials = "*string(numtrials)*"\n"
        logstr *= "\n"*readme*"\n"

        logfile = destination*"_log.txt"
        touch(logfile)
        io = open(logfile, "w")
        write(io, logstr)
        close(io)

        fprintln(logstr)

        # setting up the kernel function

        if kernel == "gaussian"
            kfunc = x -> exp(-.5*(x/bandwidth)^2)
        elseif kernel == "laplace"
            kfunc = x -> exp(-x/bandwidth)
        elseif kernel == "inv-1"
            kfunc = x -> 1/(1 + x/bandwidth)
        elseif kernel == "inv-2"
            kfunc = x -> 1/(1 + (x/bandwidth)^2)
        else
            throw(ArgumentError("unrecognized kernel '"*kernel*"'"))
        end

        # setting up arrays to record data

        cceqr_cycles = zeros(length(n_points))
        cceqr_avgblk = zeros(length(n_points))
        cceqr_active = zeros(length(n_points))

        # setting up scratch spaces

        N         = n_points[end]
        k         = n_centers*n_shells
        X_full    = zeros(N, 2*k)
        tmp_full  = zeros(N, 2*k)
        
        fprintln("\ngenerating dataset...")

        data  = zeros(2, N)         # stores data points
        label = zeros(Int64, N)     # stores true labels
        
        s_wght = Weights(1:n_shells)        # weights to assign random shell indices
        rmax   = radius*sin(pi/n_centers)   # maximum shell radius

        for i = 1:N
            c_idx    = floor(Int64, n_centers*rand(rng))
            s_idx    = sample(rng, s_wght)
            label[i] = n_shells*(c_idx - 1) + s_idx

            x1        = collect(reim(exp(2*pi*im*c_idx/n_centers)))
            x2        = collect(reim(exp(2*pi*im*rand(rng))))
            r2        = .9*rmax*s_idx/n_shells
            data[:,i] = radius*x1 + r2*x2 + noise*randn(rng, 2)
        end

        for (n_idx, n) in enumerate(n_points)
            fprintln("\nDATASET SIZE "*string(n_idx)*" OF "*string(length(n_points)))
            fprintln("-------------------------------------")
            fprintln("    selecting data points...")

            s = randperm(rng, N)[1:n]

            fprintln("    estimating kernel matrix eigenvectors...")

            # rank-2k approximation of kernel evaluation matrix using
            # randomly pivoted partial Cholesky

            X = view(X_full, 1:n, :)
            rpchol!(rng, 2*k, data, s, kfunc, X)

            # approximate eigenvalue decomposition

            svdobj = svd(X'*X)
            tmp    = view(tmp_full, 1:n, :)

            mul!(tmp, X, svdobj.V)
            rmul!(tmp, inv(Diagonal(sqrt.(svdobj.S))))
            
            V = view(tmp, :, 1:k)

            fprintln("\n    running CPQR clustering...")

            A = zeros(k, n)

            # precompiling CPQR algorithms

            copy!(A, V')
            p_geqp3 = qr!(A, ColumnNorm()).p[1:k]
            copy!(A, V')
            p_cceqr, blocks, avg_b, act = cceqr!(A, eta = eta, rho = rho)
            
            # making sure the "homemade" CPQR is giving the right results
            
            if(p_geqp3[1:k] != p_cceqr)
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                copy!(A, V')
                @save destination*"_failure_data.jld2" A j expected got
                throw(error("incorrect permutation from cceqr"))
            end

            cceqr_cycles[n_idx] = blocks
            cceqr_avgblk[n_idx] = avg_b/n
            cceqr_active[n_idx] = act/n

            @save destination*"_data.jld2" cceqr_cycles cceqr_avgblk cceqr_active
        end
    end

    run_cluster_experiment(rng, n_centers, n_shells, radius, noise,
                           n_points, kernel, bandwidth, eta, rho, numtrials,
                           plot_only, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################