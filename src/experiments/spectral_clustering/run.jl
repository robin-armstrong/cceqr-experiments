using LinearAlgebra
using CairoMakie
using StatsBase
using Debugger
using Random
using JLD2

include("../../algorithms/cceqr.jl")
include("../../algorithms/rpchol.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng = MersenneTwister(1)

n_centers = 3
n_shells  = 2
radius    = .01
noise     = .0001
n_points  = 2*round.(Int64, exp10.(range(3, 3, 1)))     # range of dataset sizes

kernel    = "inv-1"    # type of kernel function      
bandwidth = .0005          # bandwidth of kernel function

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
    function fprint(s)
        print(s)
        flush(stdout)
    end

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

        cceqr_cycles  = zeros(length(n_points))
        cceqr_avgblk  = zeros(length(n_points))
        cceqr_active  = zeros(length(n_points))
        cluster_skill = zeros(length(n_points))

        # setting up scratch spaces

        N          = n_points[end]
        k          = n_centers*n_shells
        X_full     = zeros(N, 10*k)
        tmp_full   = zeros(N, 10*k)
        dummyrange = Array(1:N)
        
        fprintln("generating dataset...")

        data   = zeros(2, N)         # stores data points
        labels = zeros(Int64, N)     # stores true labels
        
        s_wght = Weights(1:n_shells)        # weights to assign random shell indices
        rmax   = radius*sin(pi/n_centers)   # maximum shell radius

        for i = 1:N
            c_idx     = floor(Int64, n_centers*rand(rng))
            s_idx     = sample(rng, s_wght)
            labels[i] = n_shells*c_idx + s_idx

            x1        = collect(reim(exp(2*pi*im*c_idx/n_centers)))
            x2        = collect(reim(exp(2*pi*im*rand(rng))))
            r2        = .5*rmax*s_idx/n_shells
            data[:,i] = radius*x1 + r2*x2 + noise*randn(rng, 2)
        end

        fprintln("    estimating adjacency matrix degrees...")

        degrees = zeros(N)

        # we're going to populate `degrees` block-by-block, where
        # each block corresponds to data points with the same
        # label. For each such block we compute the average
        # degree of a small random sample of points, and use
        # this as the approximate degree for every point in the
        # block. The intuition here is that due to the geometric
        # symmetry of the problem, all points with the same label
        # should have more-or-less the same degree.

        for c_idx = 0:(n_centers - 1)
            for s_idx = 1:n_shells
                l      = n_shells*c_idx + s_idx
                l_idxs = dummyrange[labels .== l]

                sampsize = min(length(l_idxs), 100)
                samp     = l_idxs[randperm(rng, length(l_idxs))[1:sampsize]]
                K        = zeros(sampsize, N)

                for i = 1:sampsize
                    for j = 1:N
                        K[i,j] = kfunc(norm(data[:, samp[i]] - data[:, j]))
                    end
                end

                deg    = mean(K*ones(N))
                d_view = view(degrees, l_idxs)
                fill!(d_view, deg)
            end
        end

        #### BEGIN DEBUGGING BLOCK
            CairoMakie.activate!(visible = false, type = "pdf")
            fig = Figure(size = (700, 700))
            plt = Axis(fig[1,1])
            scatter!(plt, data)
            save(destination*"_plot.pdf", fig)

            K_full = zeros(N, N)

            for i = 1:N
                for j = i:N
                    K_full[i,j] = kfunc(norm(data[:,i] - data[:,j]))
                    K_full[j,i] = K_full[i,j]
                end
            end

            true_degrees = K_full*ones(N)
            D_sqrt       = Diagonal(sqrt.(true_degrees))

            rmul!(K_full, inv(D_sqrt))
            lmul!(inv(D_sqrt), K_full)

            fprintln("    full SVD of kernel matrix :( ...")
            true_svd = svd(K_full)
        #### END DEBUGGING BLOCK

        for (n_idx, n) in enumerate(n_points)
            fprintln("\nDATASET SIZE "*string(n_idx)*" OF "*string(length(n_points)))
            fprintln("-------------------------------------")
            fprintln("    selecting data points...")

            samp = randperm(rng, N)[1:n]

            fprintln("    approximating normalized adjacency matrix...")

            # rank-2k approximation of kernel evaluation matrix using
            # randomly pivoted partial Cholesky

            X = view(X_full, 1:n, :)
            rpchol!(rng, 10*k, data, degrees, samp, kfunc, X)

            # approximate eigenvalue decomposition

            s1  = svd(X'*X)
            tmp = view(tmp_full, 1:n, :)

            mul!(tmp, X, s1.V)
            rmul!(tmp, inv(Diagonal(sqrt.(s1.S))))
            
            V = view(tmp, :, 1:k)

            #### BEGIN DEBUGGING BLOCK
                deg_err = norm(degrees - true_degrees)/norm(true_degrees)
                fprintln("\ndegree estimation errors = "*string(deg_err))

                fprint("\nkernel matrix spectrum (1:(2k+1)) = ")
                display(true_svd.S[1:(2*k+1)])
                gamma = true_svd.S[k+1]/true_svd.S[k]
                fprintln("\nspectral gap = "*string(gamma))

                err_true = norm(K_full[samp, samp] - X*X')
                K_norm   = norm(true_svd.S)
                fprintln("\nrpchol relative error = "*string(err_true/K_norm))

                cosines = svd(true_svd.Vt[1:k, :]*V).S
                fprint("\ncosines = ")
                display(cosines)
                return
            #### END DEBUGGING BLOCK

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

            # finding approximate cone centers in embedded space
            s2 = svd(V[p_geqp3, :]')
            Q  = s2.U*s2.Vt

            # classifying the data
            mul!(A, Q, V')
            broadcast!(abs, A, A)
            learned_labels = argmax(A, dims = 1)

            fprintln("    finding learned/true labels correspondence...")

            match_table = zeros(k, k)

            for i = 1:n
                l = learned_labels[i][1]
                match_table[l, labels[samp[i]]] += 1
            end

            label_perm = zeros(Int64, k)

            for l = 1:k
                idx                = findmax(match_table)[2]
                label_perm[idx[1]] = idx[2]

                fill!(view(match_table, idx[1], :), -1.)
                fill!(view(match_table, :, idx[2]), -1.)
            end

            fprintln("    measuring cluster accuracy...")

            for i = 1:n
                l = learned_labels[i][1]
                
                if(label_perm[l] == labels[samp[i]])
                    cluster_skill[n_idx] += 1/n
                end
            end

            @save destination*"_data.jld2" cceqr_cycles cceqr_avgblk cceqr_active cluster_skill
        end
    end

    run_cluster_experiment(rng, n_centers, n_shells, radius, noise,
                           n_points, kernel, bandwidth, eta, rho, numtrials,
                           plot_only, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" cceqr_cycles cceqr_avgblk cceqr_active cluster_skill

println("CLUSTER SKILL:")
display(cluster_skill)