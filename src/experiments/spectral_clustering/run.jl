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

k        = 2
scale    = 10.
noise    = .5
n_points = 4*round.(Int64, exp10.(range(3, 3, 1)))     # range of dataset sizes

kernel    = "inv-1"  # type of kernel function      
bandwidth = .5         # bandwidth of kernel function

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

    function run_cluster_experiment(rng, k, noise, scale, n_points,
                                    kernel, bandwidth, eta, rho, numtrials,
                                    plot_only, destination, readme)
        
        logstr  = "rng       = "*string(rng)*"\n"
        logstr *= "k         = "*string(k)*"\n"
        logstr *= "noise     = "*string(noise)*"\n"
        logstr *= "scale     = "*string(scale)*"\n"
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

        data    = zeros(k, N)       # stores data points
        # labels  = zeros(Int64, N)   # stores true labels
        sqnorms = zeros(N)
        
        for j = 1:N
            i          = rand(rng, 1:k)
            data[i,j]  = scale
            data[:,j] += noise*randn(rng, k)
            sqnorms[j] = norm(data[:,j])^2
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

            # classifying the data into mixture components

            mul!(A, Q, V')
            broadcast!(abs, A, A)
            labels_raw = argmax(A, dims = 1)
            labels     = [labels_raw[i][1] for i = 1:n]

            fprintln("    measuring cluster accuracy...")

            means = zeros(k, k)
            vars  = zeros(k)

            for i = 1:k
                idx        = (labels .== i)
                num_idx    = length(idx)

                # finding the mean and variance for the given mixture component

                means[:,i] = mean(data[:,samp[idx]], dims = 2)
                
                for j = 1:n
                    idx[j]  || continue
                    vars[i] += norm(data[:,samp[j]] - means[:,i])^2/(num_idx - 1)
                end

                # multivariate Gaussian pdf for this mixture component
                p(x) = exp(-.5*norm(x - means[:,i])^2/vars[i])/(sqrt(2*pi*vars[i])^k)
                
                # E[p(X)] where X ~ Gaussian(means[i], vars[i]*I)
                E = (.5/sqrt(pi*vars[i]))^k

                for j = 1:n
                    idx[j]  || continue
                    cluster_skill[n_idx] += p(data[:,samp[j]])/(n*E)
                end
            end

            ### BEGIN DEBUGGING BLOCK
                CairoMakie.activate!(visible = false, type = "pdf")
                fig = Figure(size = (700, 700))
                plt = Axis(fig[1,1])

                for i = 1:k
                    idx = (labels .== i)
                    scatter!(plt, data[:, samp[idx]])
                end

                save(destination*"_plot.pdf", fig)
            ### END DEBUGGING BLOCK

            @save destination*"_data.jld2" cceqr_cycles cceqr_avgblk cceqr_active cluster_skill means vars
        end
    end

    run_cluster_experiment(rng, k, noise, scale, n_points, kernel, bandwidth, eta, rho, numtrials,
                           plot_only, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" cceqr_cycles cceqr_avgblk cceqr_active cluster_skill means vars

println("CLUSTER SKILL:")
display(cluster_skill)