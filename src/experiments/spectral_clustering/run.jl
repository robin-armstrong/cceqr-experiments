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

rng = MersenneTwister(2)

k        = 4
noise    = .05
n_points = 4*round.(Int64, exp10.(range(3, 3, 1)))     # range of dataset sizes

kernel    = "gaussian"  # type of kernel function      
bandwidth = .1         # bandwidth of kernel function

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

    function run_cluster_experiment(rng, k, noise, n_points, kernel,
                                    bandwidth, eta, rho, numtrials,
                                    plot_only, destination, readme)
        
        logstr  = "rng       = "*string(rng)*"\n"
        logstr *= "k         = "*string(k)*"\n"
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
            kfunc = x -> exp(-(x/bandwidth)^2)
        elseif kernel == "laplace"
            kfunc = x -> exp(-x/bandwidth)
        elseif kernel == "inv-1"
            kfunc = x -> 1/(1 + x/bandwidth)
        elseif kernel == "inv-2"
            kfunc = x -> 1/(1 + (x/bandwidth)^2)
        elseif kernel == "window"
            kfunc = x -> (x <= bandwidth ? 1. : 0.)
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
        X_full     = zeros(N, 10*k)
        tmp_full   = zeros(N, 10*k)
        dummyrange = Array(1:N)
        
        fprintln("generating dataset...")

        data    = zeros(2, N)       # stores data points
        labels  = zeros(Int64, N)   # stores true labels
        sqnorms = zeros(N)
        w       = Weights(1:k)
        
        for i = 1:N
            l          = rand(rng, 1:k)
            labels[i]  = l

            t = 2*pi*rand(rng)
            
            while rand(rng) > sqrt(.5*(1 + cos(t)^2))
                t = 2*pi*rand(rng)
            end

            data[:,i]  = [1.5*pi*l + t, sin(t)] + noise*randn(rng, 2)
            sqnorms[i] = norm(data[:,i])^2
        end

        fprintln("calculating adjacency matrix degrees...")

        degrees = zeros(N)
        tmp     = zeros(N)

        for i = 1:N
            fill!(tmp, sqnorms[i])

            tmp .+= sqnorms
            tmp .-= 2*data'*data[:, i]

            broadcast!(abs, tmp, tmp)
            broadcast!(sqrt, tmp, tmp)
            broadcast!(kfunc, tmp, tmp)

            degrees[i] = sum(tmp)
        end

        for (n_idx, n) in enumerate(n_points)
            fprintln("\nDATASET SIZE "*string(n_idx)*" OF "*string(length(n_points)))
            fprintln("-------------------------------------")
            fprintln("    selecting data points...")

            samp = 1:N #randperm(rng, N)[1:n]

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

            fprintln("\n    running CPQR clustering...")

            A = zeros(k, n)

            # precompiling CPQR algorithms

            copy!(A, V')
            p_geqp3 = qr!(A, ColumnNorm()).p[1:k]
            copy!(A, V')
            p_cceqr, blocks, avg_b, act = cceqr!(A, eta = eta, rho = rho)

            println("p_geqp3[1:k] = ")
            display(p_geqp3[1:k])

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
            learned_labels_raw = argmax(A, dims = 1)
            learned_labels     = [learned_labels_raw[i][1] for i = 1:n]

            ### BEGIN DEBUGGING BLOCK
                CairoMakie.activate!(visible = false, type = "pdf")
                fig = Figure(size = (1200, 600))
                plt = Axis(fig[1,1])

                for l = 1:k
                    label_idx = (learned_labels .== l)
                    scatter!(plt, data[:,label_idx])
                end

                save(destination*"_plot.pdf", fig)
            ### END DEBUGGING BLOCK

            fprintln("    finding learned/true labels correspondence...")

            match_table = zeros(k, k)

            for i = 1:n
                l = learned_labels[i]
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
                l = learned_labels[i]
                
                if(label_perm[l] == labels[samp[i]])
                    cluster_skill[n_idx] += 1/n
                end
            end

            @save destination*"_data.jld2" cceqr_cycles cceqr_avgblk cceqr_active cluster_skill data A labels learned_labels label_perm
        end
    end

    run_cluster_experiment(rng, k, noise, n_points, kernel, bandwidth, eta, rho, numtrials,
                           plot_only, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" cceqr_cycles cceqr_avgblk cceqr_active cluster_skill data A labels learned_labels label_perm

println("CLUSTER SKILL:")
display(cluster_skill)