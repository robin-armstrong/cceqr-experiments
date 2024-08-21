using LinearAlgebra
using StatsBase
using Random
using PyPlot
using JLD2

include("../../algorithms/tcpqr.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng = MersenneTwister(1)

d      = 100                # data dimension
k      = 20                 # number of mixture components, should be less than d
n      = 20000              # k*n samples will be drawn from the mixture
srange = range(1, 10, 10)   # cluster separation values

kernel           = "inv-1"  # type of kernel function      
bandwidth        = 1.       # bandwidth of kernel function
kernel_blocksize = 20000    # used for computing kernel matrix eigenvectors, should be a divisor of k*n

mu               = 0.9      # controls threshold value for TCPQR
numtrials        = 100      # algorithm trials per separation value

plot_only           = false     # if "true" then data will be read from disk and not regenerated
generate_embeddings = true      # if "true" then embeddings will be calculated from scratch, otherwise read from disk
embedding_name      = "src/experiments/spectral_clustering/huge_embeddings.jld2"

destination = "src/experiments/spectral_clustering/hugetest"
readme      = "Comparing algorithm performance on a GIGANTIC example!"

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

if(!plot_only)
    # logging the parameters used for this experiment

    function fprintln(s)
        println(s)
        flush(stdout)
    end

    function kernel_product!(X, Y, data, s, sqnorms, kernel, bandwidth, kernel_blocksize, tmp)
        n = size(data, 2)
        p = size(X, 2)

        if(n % kernel_blocksize != 0)
            throw(error("number of data points must be a multiple of the kernel_blocksize"))
        elseif(size(Y) != (n, p))
            throw(error("incorrect dimensions for product matrix"))
        end

        if(kernel == "gaussian")
            kfunc = x -> exp(-.5*(x/bandwidth)^2/d)
        elseif(kernel == "laplace")
            kfunc = x -> exp(-x/(sqrt(d)*bandwidth))
        elseif(kernel == "inv-1")
            kfunc = x -> sqrt(d)/(1 + x/bandwidth)
        elseif(kernel == "inv-2")
            kfunc = x -> d/(1 + (x/bandwidth)^2)
        else
            throw(ArgumentError("unrecognized kernel '"*kernel*"'"))
        end

        nblocks = div(n, kernel_blocksize)
        e1      = ones(n)
        e2      = ones(kernel_blocksize)
        fill!(Y, 0.)
        
        for i = 1:nblocks
            fprintln("            KERNEL PRODUCT: block "*string(i)*" of "*string(nblocks))
            # constructing columns kernel_blocksize*(i - 1) + 1 through kernel_blocksize*i of the kernel evaluation matrix, storing to tmp
            fill!(tmp, 0.)

            block     = (kernel_blocksize*(i - 1) + 1):(kernel_blocksize*i)
            tmp[:,:] += sqnorms[s, :]*e2'
            tmp[:,:] += e1*sqnorms[s, block]'
            tmp[:,:] -= 2*data'*data[:, block]

            broadcast!(abs, tmp, tmp)
            broadcast!(sqrt, tmp, tmp)
            broadcast!(kfunc, tmp, tmp)

            # computing a set of kernel_blocksize outer products that contribute to the final matrix-matrix product
            Y[:, :] += tmp*X[block, :]
        end
    end

    function kernel_rsvd!(rng, data, s, sqnorms, kernel, bandwidth, kernel_blocksize, p, q, V, tmp1, tmp2)
        n = size(data, 2)

        # sketching
        copy!(tmp1, randn(rng, n, p))
        
        for i = 1:q
            fprintln("        RSVD: kernel product for power iteration "*string(i))
            kernel_product!(tmp1, V, data, s, sqnorms, kernel, bandwidth, kernel_blocksize, tmp2)
            qrobj = qr!(V)

            # writing the orthogonalized sketch to scratch2
            fill!(tmp1, 0.)
            for i = 1:p tmp1[i, i] = 1. end
            lmul!(qrobj.Q, tmp1)
        end

        # projection, then SVD
        fprintln("        RSVD: final kernel product")
        kernel_product!(tmp1, V, data, s, sqnorms, kernel, bandwidth, kernel_blocksize, tmp2)
        V_small, _, _ = svd(tmp1'*V)

        mul!(V, tmp1, V_small)
    end

    function run_experiment(rng, d, k, n, srange, kernel, bandwidth, kernel_blocksize, mu, numtrials, plot_only, generate_embeddings, embedding_name, destination, readme)
        if(generate_embeddings)
            V           = zeros(k*n, k + 10)
            tmp1        = zeros(k*n, k + 10)
            tmp2        = zeros(k*n, kernel_blocksize)
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
        logstr *= "kernel_blocksize    = "*string(kernel_blocksize)*"\n"
        logstr *= "mu                  = "*string(mu)*"\n"
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

        tcpqr_time    = zeros(length(srange), numtrials)
        tcpqr_cycles  = zeros(length(srange))
        tcpqr_avgblk  = zeros(length(srange))
        tcpqr_active  = zeros(length(srange))
        geqp3_time    = zeros(length(srange), numtrials)
        col_imbalance = zeros(length(srange))
        avg_angle     = zeros(length(srange))
        cluster_skill = zeros(length(srange))
        scratch       = zeros(k, k*n)

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

                kernel_rsvd!(rng, data, s, sqnorms, kernel, bandwidth, kernel_blocksize, k + 10, 3, V, tmp1, tmp2)

                embedding[:, :, s] = V[:, 1:k]'

                fprintln("\n   measuring column norms...")

                for i = 1:(k*n)
                    sqnorms[s, i] = norm(embedding[:, i, s])^2
                end

                @save embedding_name embedding sqnorms true_labels srange k n
            end

            col_imbalance[s] = sum(sort(sqnorms[s, :], rev = true)[1:k])/k

            fprintln("   running CPQR clustering...")
            
            # precompiling CPQR algorithms
            copy!(scratch, embedding[:, :, s])
            p_geqp3 = qr!(scratch, ColumnNorm()).p[1:k]
            copy!(scratch, embedding[:, :, s])
            p_tcpqr, blocks, avg_b, act = tcpqr!(scratch, mu = mu)

            # making sure the "homemade" CPQRs are giving the right results

            if(p_geqp3[1:k] != p_tcpqr)
                j = 1
                while(p_geqp3[j] == p_tcpqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_tcpqr[j]

                @save destination*"_failure_data.jld2" embedding j expected got
                throw(error("incorrect permutation from tcpqr"))
            end

            tcpqr_cycles[s] = blocks
            tcpqr_avgblk[s] = avg_b/(k*n)
            tcpqr_active[s] = act/(k*n)

            # finding approximate cone centers in embedded space
            svdobj = svd(embedding[:, p_geqp3, s])
            Q      = svdobj.U*svdobj.Vt

            # measuring angles between embedded data and cone centers
            mul!(scratch, Q', embedding[:, :, s])
            rmul!(scratch, Diagonal(sqnorms[s, :].^(-.5)))

            # classifying the data
            labels = argmax(abs.(scratch), dims = 1)

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
                avg_angle[s] += acos(abs(scratch[l, i]))
                
                if(label_perm[l] == true_labels[s, i]) cluster_skill[s] += 1 end
            end

            avg_angle[s]     /= k*n
            cluster_skill[s] /= k*n

            fprintln("   measuring CPQR runtimes...")
            
            for trial = 1:numtrials
                fprintln("       trial "*string(trial)*" of "*string(numtrials)*"...")

                copy!(scratch, embedding[:, :, s])
                t = @elapsed qr!(scratch, ColumnNorm())
                geqp3_time[s, trial] = t

                copy!(scratch, embedding[:, :, s])
                t = @elapsed tcpqr!(scratch, mu = mu)
                tcpqr_time[s, trial] = t
            end

            fprintln("")

            @save destination*"_data.jld2" srange geqp3_time tcpqr_time tcpqr_cycles tcpqr_avgblk tcpqr_active col_imbalance avg_angle cluster_skill
        end
    end

    run_experiment(rng, d, k, n, srange, kernel, bandwidth, kernel_blocksize, mu, numtrials, plot_only, generate_embeddings, embedding_name, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" srange geqp3_time tcpqr_time tcpqr_cycles tcpqr_avgblk tcpqr_active col_imbalance avg_angle cluster_skill

ioff()
fig = figure(figsize = (6, 22))

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
time.plot(srange, vec(mean(tcpqr_time, dims = 2))*1000, color = "brown", marker = "v", markerfacecolor = "none", label = "TCPQR")
time.legend()

cycles.set_xlabel("Cluster Separation Value")
cycles.set_ylabel("TCPQR Cycle Count")
cycles.plot(srange, tcpqr_cycles, color = "brown", marker = "v", markerfacecolor = "none")

block.set_xlabel("Cluster Separation Value")
block.set_ylabel("TCPQR Average Block Percentage")
block.plot(srange, tcpqr_avgblk, color = "brown", marker = "v", markerfacecolor = "none")

active.set_xlabel("Cluster Separation Value")
active.set_ylabel("TCPQR Active Set Percentage")
active.plot(srange, tcpqr_active, color = "brown", marker = "v", markerfacecolor = "none")

savefig(destination*"_plot.pdf")
close(fig)
