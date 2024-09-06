using LinearAlgebra
using StatsBase
using Random
using PyPlot
using JLD2

include("../../algorithms/cceqr.jl")
include("../../algorithms/rpchol.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng = MersenneTwister(1)

d      = 100                # data dimension
k      = 20                 # number of mixture components, should be less than d
n      = 20000              # k*n samples will be drawn from the mixture
srange = range(1, 10, 10)   # cluster separation values

kernel    = "inv-1"     # type of kernel function      
bandwidth = 1.          # bandwidth of kernel function

eta       = .1          # controls threshold value for CCEQR
rho       = 1e-4        # controls selection of columns for Householder reflection
numtrials = 100         # algorithm trials per separation value

plot_only           = false     # if "true" then data will be read from disk and not regenerated
generate_embeddings = true      # if "true" then embeddings will be calculated from Vt, otherwise read from disk
embedding_name      = "src/experiments/spectral_clustering/embeddings.jld2"

destination = "src/experiments/spectral_clustering/cceqr_test"
readme      = "Testing CCEQR on very large clustering problems."

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

if(!plot_only)
    # logging the parameters used for this experiment

    function fprintln(s)
        println(s)
        flush(stdout)
    end

    function run_cluster_experiment(rng, d, k, n, srange, kernel, bandwidth, eta, rho, numtrials, plot_only, generate_embeddings, embedding_name, destination, readme)
        if(generate_embeddings)
            X           = zeros(k*n, 2*k)
            tmp         = zeros(k*n, 2*k)
            data        = zeros(d, k*n)
            embedding   = zeros(k, k*n, length(srange))
            sqnorms     = zeros(length(srange), k*n)
            true_labels = zeros(Int64, length(srange), k*n)
        else
            fprintln("loading embeddings...\n")
            @load embedding_name embedding sqnorms true_labels srange k n
        end
        
        logstr  = "rng                 = "*string(rng)*"\n"
        logstr *= "d                   = "*string(d)*"\n"
        logstr *= "k                   = "*string(k)*"\n"
        logstr *= "n                   = "*string(n)*"\n"
        logstr *= "srange              = "*string(srange)*"\n"
        logstr *= "kernel              = "*kernel*"\n"
        logstr *= "bandwidth           = "*string(bandwidth)*"\n"
        logstr *= "eta                 = "*string(eta)*"\n"
        logstr *= "rho                 = "*string(rho)*"\n"
        logstr *= "numtrials           = "*string(numtrials)*"\n"
        logstr *= "generate_embeddings = "*string(generate_embeddings)*"\n"
        logstr *= "embedding_name      = "*embedding_name*"\n"
        logstr *= "\n"*readme*"\n"

        logfile = destination*"_log.txt"
        touch(logfile)
        open(logfile, "w")
        write(logfile, logstr)
        close(logfile)

        fprintln(logstr)

        cceqr_time    = zeros(length(srange), numtrials)
        cceqr_cycles  = zeros(length(srange))
        cceqr_avgblk  = zeros(length(srange))
        cceqr_active  = zeros(length(srange))
        geqp3_time    = zeros(length(srange), numtrials)
        col_imbalance = zeros(length(srange))
        avg_angle     = zeros(length(srange))
        cluster_skill = zeros(length(srange))
        Vt            = zeros(k, k*n)

        for s = 1:length(srange)
            fprintln("   SCALE FACTOR "*string(s)*" of "*string(length(srange)))
            fprintln("   ------------")
            
            if(generate_embeddings)
                fprintln("   generating data...")
                
                scale = srange[s]
                
                for i = 1:(k*n)
                    l                 = rand(rng, 1:k)  # random mixture component to draw from
                    true_labels[s, i] = l
                    data[:, i]        = randn(rng, d)
                    data[l, i]       += scale
                    sqnorms[s, i]     = norm(data[:,i])^2
                end

                fprintln("   estimating kernel matrix eigenvectors...")

                if kernel == "gaussian"
                    kfunc = x -> exp(-.5*(x/bandwidth)^2/d)
                elseif kernel == "laplace"
                    kfunc = x -> exp(-x/(sqrt(d)*bandwidth))
                elseif kernel == "inv-1"
                    kfunc = x -> sqrt(d)/(1 + x/bandwidth)
                elseif kernel == "inv-2"
                    kfunc = x -> d/(1 + (x/bandwidth)^2)
                else
                    throw(ArgumentError("unrecognized kernel '"*kernel*"'"))
                end

                fill!(X, 0.)
                fill!(tmp, 0.)

                # rank-2k approximation using randomly pivoted partial Cholesky
                rpchol!(rng, 2*k, data, kfunc, X)

                # approximate eigenvalue decomposition
                svdobj = svd(X'*X)
                mul!(tmp, X, svdobj.V)
                rmul!(tmp, inv(Diagonal(sqrt.(svdobj.S))))
                embedding[:, :, s] = tmp[:, 1:k]'

                fprintln("\n   measuring column norms...")

                for i = 1:(k*n)
                    sqnorms[s, i] = norm(embedding[:, i, s])^2
                end

                @save embedding_name embedding sqnorms true_labels srange k n
            end

            col_imbalance[s] = sum(sort(sqnorms[s, :], rev = true)[1:k])/k

            fprintln("   running CPQR clustering...")
            
            # precompiling CPQR algorithms
            copy!(Vt, embedding[:, :, s])
            p_geqp3 = qr!(Vt, ColumnNorm()).p[1:k]
            copy!(Vt, embedding[:, :, s])
            p_cceqr, blocks, avg_b, act = cceqr!(Vt, eta = eta, rho = rho)

            # making sure the "homemade" CPQRs are giving the right results

            if(p_geqp3[1:k] != p_cceqr)
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                copy!(Vt, embedding[:, :, s])
                @save destination*"_failure_data.jld2" Vt j expected got
                throw(error("incorrect permutation from cceqr"))
            end

            cceqr_cycles[s] = blocks
            cceqr_avgblk[s] = avg_b/(k*n)
            cceqr_active[s] = act/(k*n)

            # finding approximate cone centers in embedded space
            svdobj = svd(embedding[:, p_geqp3, s])
            Q      = svdobj.U*svdobj.Vt

            # measuring angles between embedded data and cone centers
            mul!(Vt, Q', embedding[:, :, s])
            rmul!(Vt, Diagonal(sqnorms[s, :].^(-.5)))

            # classifying the data
            labels = argmax(abs.(Vt), dims = 1)

            fprintln("   finding learned/true label correspondence...")
            
            match_table = zeros(k, k)

            for i = 1:(k*n)
                l_learned = labels[i][1]
                match_table[l_learned, true_labels[s, i]] += 1
            end

            label_perm  = zeros(Int64, k)

            for l = 1:k
                idx                = findmax(match_table)[2]
                label_perm[idx[1]] = idx[2]

                fill!(view(match_table, idx[1], :), -1.)
                fill!(view(match_table, :, idx[2]), -1.)
            end

            fprintln("   measuring clustering skill and OCS...")

            for i = 1:(k*n)
                l             = labels[i][1]
                avg_angle[s] += acos(abs(Vt[l, i]))
                
                if(label_perm[l] == true_labels[s, i]) cluster_skill[s] += 1 end
            end

            avg_angle[s]     /= k*n
            cluster_skill[s] /= k*n

            fprintln("   measuring CPQR runtimes...")
            
            for trial = 1:numtrials
                fprintln("       trial "*string(trial)*" of "*string(numtrials)*"...")

                copy!(Vt, embedding[:, :, s])
                t = @elapsed qr!(Vt, ColumnNorm())
                geqp3_time[s, trial] = t

                copy!(Vt, embedding[:, :, s])
                t = @elapsed cceqr!(Vt, eta = eta, rho = rho)
                cceqr_time[s, trial] = t
            end

            fprintln("")

            @save destination*"_data.jld2" srange geqp3_time cceqr_time cceqr_cycles cceqr_avgblk cceqr_active col_imbalance avg_angle cluster_skill
        end
    end

    run_cluster_experiment(rng, d, k, n, srange, kernel, bandwidth, eta, rho, numtrials, plot_only, generate_embeddings, embedding_name, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" srange geqp3_time cceqr_time cceqr_cycles cceqr_avgblk cceqr_active col_imbalance avg_angle cluster_skill

ioff()
fig = figure(figsize = (7, 22))

skill  = fig.add_subplot(5, 1, 1)
time   = fig.add_subplot(5, 1, 2)
cycles = fig.add_subplot(5, 1, 3)
block  = fig.add_subplot(5, 1, 4)
active = fig.add_subplot(5, 1, 5)

skill.set_xlabel("Cluster Separation Value")
skill.set_ylabel("Percentage of Correct Assignments")
skill.set_ylim([-.05, 1.05])
skill.plot(srange, cluster_skill, color = "black", marker = "s", markerfacecolor = "none")

time.set_xlabel("Cluster Separation Value")
time.set_ylabel("Running Time (ms)")
time.axhline(mean(geqp3_time)*1000, color = "blue", linestyle = "dashed", label = "GEQP3")
time.plot(srange, vec(mean(cceqr_time, dims = 2))*1000, color = "brown", marker = "v", markerfacecolor = "none", label = "CCEQR")
time.legend()

cycles.set_xlabel("Cluster Separation Value")
cycles.set_ylabel("CCEQR Cycle Count")
cycles.plot(srange, cceqr_cycles, color = "brown", marker = "v", markerfacecolor = "none")

block.set_xlabel("Cluster Separation Value")
block.set_ylabel("CCEQR Average Block Percentage")
block.plot(srange, cceqr_avgblk, color = "brown", marker = "v", markerfacecolor = "none")

active.set_xlabel("Cluster Separation Value")
active.set_ylabel("CCEQR Active Set Percentage")
active.plot(srange, cceqr_active, color = "brown", marker = "v", markerfacecolor = "none")

savefig(destination*"_plot.pdf")
close(fig)
