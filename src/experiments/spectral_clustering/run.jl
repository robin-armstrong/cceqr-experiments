using LinearAlgebra
using CairoMakie
using StatsBase
using Debugger
using Random
using JLD2

include("../../algorithms/cceqr.jl")
include("../../algorithms/pivchol.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng = MersenneTwister(2)

n      = 400000    # number of data points, must be > 4*k
k      = 20        # number of Gaussian mixture components
srange = 1:10
noise  = 1.

kernel    = "gaussian"  # type of kernel function      
bandwidth = 10.         # bandwidth of kernel function

rho_range = exp10.(range(-5, -.3, 10))
numtrials = 10

plot_only   = false       # if "true" then data will be read from disk and not regenerated
destination = "src/experiments/spectral_clustering/cluster"
readme      = "Comparing GEQP3 and CCEQR on a spectral clustering problem."

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

    function run_cluster_experiment(rng, n, k, srange, noise, kernel, bandwidth,
                                    rho_range, numtrials, destination, readme)
        
        logstr  = "rng       = "*string(rng)*"\n"
        logstr *= "n         = "*string(n)*"\n"
        logstr *= "k         = "*string(k)*"\n"
        logstr *= "srange    = "*string(srange)*"\n"
        logstr *= "noise     = "*string(noise)*"\n"
        logstr *= "kernel    = "*kernel*"\n"
        logstr *= "bandwidth = "*string(bandwidth)*"\n"
        logstr *= "srange    = "*string(srange)*"\n"
        logstr *= "rho_range = "*string(rho_range)*"\n"
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

        cceqr_cycles = zeros(length(srange), length(rho_range))
        cceqr_avgblk = zeros(length(srange), length(rho_range))
        cceqr_active = zeros(length(srange), length(rho_range))
        cceqr_time   = zeros(length(srange), length(rho_range), numtrials)
        geqp3_time   = zeros(numtrials)

        # preallocating some arrays for setting up the clustering problem
        data    = zeros(n, k)
        Vt      = zeros(k, n)
        tmp     = zeros(k, n)
        X       = zeros(n, 4*k)
        scratch = zeros(n, 4*k)

        totaltrials = length(rho_range)*length(srange)*numtrials
        trialcount  = 0

        for (scale_idx, scale) in enumerate(srange)
            fprintln("\n-------------------------------------------------")
            fprintln("SEPARATION VALUE "*string(scale_idx)*" OF "*string(length(srange)))
            fprintln("-------------------------------------------------\n")
            fprintln("generating dataset...")
            
            for i = 1:n
                j          = rand(rng, 1:k)
                data[i,j]  = scale
                data[i,:] += noise*randn(rng, k)
            end

            fprintln("estimating adjacency matrix eigenvectors...\n")

            pivchol!(4*k, data, kfunc, X)
            s1 = svd(X'*X)
            mul!(scratch, X, s1.V)
            rmul!(scratch, inv(Diagonal(sqrt.(s1.S))))

            copyto!(Vt, CartesianIndices((1:k, 1:n)), scratch', CartesianIndices((1:k, 1:n)))

            fprintln("\nclustering data and benchmarking GEQP3...")

            copy!(tmp, Vt)
            p_geqp3 = qr!(tmp, ColumnNorm()).p[1:k]

            for t = 1:numtrials
                copy!(tmp, Vt)
                geqp3_time[t] = @elapsed qr(tmp, ColumnNorm())
            end

            s2 = svd(Vt[:, p_geqp3])
            Q  = s2.U*s2.Vt
            W  = Q'*Vt

            labels_raw = argmax(W, dims = 1)
            labels     = [labels_raw[i][1] for i = 1:n]

            fprintln("fitting Gaussian mixture model...")

            means = zeros(k, k)
            vars  = zeros(k)

            cluster_skill = 0.

            for i = 1:k
                idx        = (labels .== i)
                num_idx    = length(idx)

                # finding the mean and variance for the given mixture component

                means[:,i] = mean(data[idx, :], dims = 1)
                
                for j = 1:n
                    idx[j]  || continue
                    vars[i] += norm(data[j,:] - means[:,i])^2/(num_idx - 1)
                end
            end

            fprintln("measuring cluster skill...")

            # learned probability density for i^th Gaussian
            p(x,i) = exp(-.5*norm(x - means[:,i])^2/vars[i])/(sqrt(2*pi*vars[i])^k)
            
            # E[p(X, i)] where X ~ Gaussian(means[:,i], vars[i]*I)
            E(i) = (.5/sqrt(pi*vars[i]))^k
            
            skill_learned = 0.
            skill_random  = 0.
            
            for i = 1:n
                l_learned = labels[i]
                l_random  = rand(rng, 1:k)

                skill_learned += p(data[i,:], l_learned)/(n*E(l_learned))
                skill_random  += p(data[i,:], l_random)/(n*E(l_random))
            end

            fprintln("skill (learned labels): "*string(skill_learned))
            fprintln("skill (random labels):  "*string(skill_random))
            fprintln("\nmeasuring CCEQR runtimes...\n")

            for (rho_idx, rho) in enumerate(rho_range)
                # precompiling CCEQR and making sure it gives the right permutation
                
                copy!(tmp, Vt)
                p_cceqr, blocks, avg_b, act = cceqr!(tmp, rho = rho)

                if(p_cceqr != p_geqp3)
                    j = 1
                    while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                    expected = p_geqp3[j]
                    got      = p_cceqr[j]

                    copy!(tmp, Vt)
                    @save destination*"_failure_data.jld2" Vt rho j expected got
                    throw(error("incorrect permutation from CCEQR"))
                end

                cceqr_cycles[scale_idx, rho_idx] = blocks
                cceqr_avgblk[scale_idx, rho_idx] = avg_b/n
                cceqr_active[scale_idx, rho_idx] = act/n

                for trial_idx = 1:numtrials
                    trialcount += 1
                    fprintln("    trial "*string(trialcount)*" of "*string(totaltrials)*"...")

                    copy!(tmp, Vt)
                    t = @elapsed cceqr!(tmp, rho = rho)
                    cceqr_time[scale_idx, rho_idx, trial_idx] = t
                end

                @save destination*"_data.jld2" n k srange rho_range skill_learned skill_random numtrials geqp3_time cceqr_time cceqr_cycles cceqr_avgblk cceqr_active
            end
        end
    end

    run_cluster_experiment(rng, n, k, srange, noise, kernel, bandwidth,
                           rho_range, numtrials, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" n k srange rho_range skill_learned skill_random numtrials geqp3_time cceqr_time cceqr_cycles cceqr_avgblk cceqr_active

cceqr_mean_times = zeros(length(srange), length(rho_range))

for i = 1:length(srange)
    for j = 1:length(rho_range)
        no_outliers = (cceqr_time[i, j, :] .< 100.)
        cceqr_mean_times[i,j] = mean(cceqr_time[i, j, no_outliers])
    end
end

geqp3_mean = mean(geqp3_time)
time_comp  = geqp3_mean*cceqr_mean_times.^(-1)

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (800, 700))

time = Axis(fig[1,1],
            title  = L"$T_\mathrm{GEQP3}/T_\mathrm{CCEQR}$",
            xlabel = L"$\log_{10} \,\rho$",
            ylabel = "Cluster Separation"
           )
heatmap!(time, log10.(rho_range), srange, time_comp)
Colorbar(fig[1,2], limits = extrema(time_comp))

blocks = Axis(fig[1,3],
              title  = "Average Block Percentage Per Cycle",
              xlabel = L"$\log_{10} \,\rho$",
              ylabel = "Cluster Separation"
             )
heatmap!(blocks, log10.(rho_range), srange, cceqr_avgblk)
Colorbar(fig[1,4], limits = extrema(cceqr_avgblk))

active = Axis(fig[2,1],
              title  = "Final Active Set Percentage",
              xlabel = L"$\log_{10} \,\rho$",
              ylabel = "Cluster Separation"
             )
heatmap!(active, log10.(rho_range), srange, cceqr_active)
Colorbar(fig[2,2], limits = extrema(cceqr_active))

cycles = Axis(fig[2,3],
              title  = "CCEQR Cycle Count",
              xlabel = L"$\log_{10} \,\rho$",
              ylabel = "Cluster Separation"
             )
heatmap!(cycles, log10.(rho_range), srange, cceqr_cycles)
Colorbar(fig[2,4], limits = extrema(cceqr_cycles))

save(destination*"_plot.pdf", fig)
