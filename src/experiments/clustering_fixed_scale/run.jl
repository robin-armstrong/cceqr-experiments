include("../../config/config.jl")

using LinearAlgebra
using CairoMakie
using StatsBase
using Debugger
using Random
using CCEQR
using JLD2

include("../../algorithms/pivchol.jl")
include("../../misc/fprints.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng = MersenneTwister(2)

nrange = round.(Int64, exp10.(range(2, 6, 9)))     # number of data points, must be > 4*k
k      = 20                                        # number of Gaussian mixture components
scale  = 6.
noise  = 1.

kernel    = "gaussian"  # type of kernel function      
bandwidth = 10.         # bandwidth of kernel function

rho_range = exp10.(range(-5, -.3, 20))
numtrials = 30

plot_only   = false     # if "true" then data will be read from disk and not regenerated
destination = "src/experiments/clustering_fixed_scale/cluster_fixedscale"
readme      = "Comparing GEQP3 and CCEQR on a spectral clustering problem."

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

function run_cluster_fixed_scale(rng, nrange, k, scale, noise, kernel, bandwidth,
                                    rho_range, numtrials, destination, readme)
    
    logstr  = "rng       = "*string(rng)*"\n"
    logstr *= "nrange    = "*string(nrange)*"\n"
    logstr *= "k         = "*string(k)*"\n"
    logstr *= "scale     = "*string(scale)*"\n"
    logstr *= "noise     = "*string(noise)*"\n"
    logstr *= "kernel    = "*kernel*"\n"
    logstr *= "bandwidth = "*string(bandwidth)*"\n"
    logstr *= "nrange    = "*string(nrange)*"\n"
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

    cceqr_cycles    = zeros(length(nrange), length(rho_range))
    cceqr_avgblk    = zeros(length(nrange), length(rho_range))
    cceqr_active    = zeros(length(nrange), length(rho_range))
    cceqr_time_cssp = zeros(length(nrange), length(rho_range), numtrials)
    cceqr_time_cpqr = zeros(length(nrange), length(rho_range), numtrials)
    geqp3_time      = zeros(length(nrange), numtrials)

    totaltrials = length(rho_range)*length(nrange)*numtrials
    trialcount  = 0

    for (n_idx, n) in enumerate(nrange)
        # preallocating some arrays for setting up the clustering problem
        data    = zeros(n, k)
        Vt      = zeros(k, n)
        tmp     = zeros(k, n)
        X       = zeros(n, 4*k)
        scratch = zeros(n, 4*k)

        fprintln("\n-------------------------------------------------")
        fprintln("DATASET SIZE "*string(n_idx)*" OF "*string(length(nrange)))
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

        for trial_idx = 1:numtrials
            copy!(tmp, Vt)
            geqp3_time[n_idx, trial_idx] = @elapsed qr(tmp, ColumnNorm())
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
            p_cceqr, blocks, avg_b, act  = cceqr!(tmp, rho = rho)
            p_cceqr                      = p_cceqr[1:k]
            cceqr_cycles[n_idx, rho_idx] = blocks
            cceqr_avgblk[n_idx, rho_idx] = avg_b
            cceqr_active[n_idx, rho_idx] = act

            if(p_cceqr != p_geqp3)
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                copy!(tmp, Vt)
                @save destination*"_failure_data.jld2" Vt rho j expected got
                throw(error("incorrect permutation from CCEQR"))
            end

            cceqr_cycles[n_idx, rho_idx] = blocks
            cceqr_avgblk[n_idx, rho_idx] = avg_b/n
            cceqr_active[n_idx, rho_idx] = act/n

            for trial_idx = 1:numtrials
                trialcount += 1
                fprintln("    trial "*string(trialcount)*" of "*string(totaltrials)*"...")

                copy!(tmp, Vt)
                t = @elapsed cceqr!(tmp, rho = rho)
                cceqr_time_cssp[n_idx, rho_idx, trial_idx] = t

                copy!(tmp, Vt)
                t = @elapsed cceqr!(tmp, rho = rho, full = true)
                cceqr_time_cpqr[n_idx, rho_idx, trial_idx] = t
            end

            @save destination*"_data.jld2" nrange k scale rho_range skill_learned skill_random numtrials geqp3_time cceqr_time_cssp cceqr_time_cpqr cceqr_cycles cceqr_avgblk cceqr_active
        end
    end
end

plot_only || run_cluster_fixed_scale(rng, nrange, k, scale, noise, kernel,
                                     bandwidth, rho_range, numtrials,
                                     destination, readme)

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" nrange k scale rho_range skill_learned skill_random numtrials geqp3_time cceqr_time_cssp cceqr_time_cpqr cceqr_cycles cceqr_avgblk cceqr_active

cceqr_median_cssp = median(cceqr_time_cssp, dims = 3)
cceqr_median_cpqr = median(cceqr_time_cpqr, dims = 3)
geqp3_median      = median(geqp3_time, dims = 2)*ones(1, length(rho_range))

cceqr_median_cssp = reshape(cceqr_median_cssp, (length(nrange), length(rho_range)))
cceqr_median_cpqr = reshape(cceqr_median_cpqr, (length(nrange), length(rho_range)))

time_comp_cssp = cceqr_median_cssp./geqp3_median
time_comp_cpqr = cceqr_median_cpqr./geqp3_median

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (600, 300), fonts = (; regular = regfont))

Lmax = 9
Lmin = exp2(-log2(9))

time_cssp = Axis(fig[1,1],
                 title       = L"$T_\mathrm{CCEQR}/T_\mathrm{GEQP3}$ (CSSP Only)",
                 xlabel      = "ρ",
                 xticks      = [-5, -4, -3, -2, -1],
                 xtickformat = xvals -> [L"10^{%$x}" for x in Int.(xvals)],
                 ylabel      = "Dataset Size",
                 yticks      = [1, 2, 3, 4, 5, 6],
                 ytickformat = yvals -> [L"10^{%$y}" for y in Int.(yvals)]
                )

heatmap!(time_cssp, log10.(rho_range), log10.(nrange), transpose(time_comp_cssp), colormap = :vik, colorscale = log2, colorrange = (Lmin, Lmax))

time_cpqr = Axis(fig[1,2],
                 title       = L"$T_\mathrm{CCEQR}/T_\mathrm{GEQP3}$ (Full CPQR)",
                 xlabel      = "ρ",
                 xticks      = [-5, -4, -3, -2, -1],
                 xtickformat = xvals -> [L"10^{%$x}" for x in Int.(xvals)],
                 ylabel      = "Dataset Size",
                 yticks      = [1, 2, 3, 4, 5, 6],
                 ytickformat = yvals -> [L"10^{%$y}" for y in Int.(yvals)]
                )

heatmap!(time_cpqr, log10.(rho_range), log10.(nrange), transpose(time_comp_cpqr), colormap = :vik, colorscale = log2, colorrange = (Lmin, Lmax))
Colorbar(fig[1,3], colormap = :vik, scale = log2, ticks = exp2.(-3:3), tickformat = vals -> [L"2^{%$i}" for i in Int.(log2.(vals))], limits = (Lmin, Lmax))

save(destination*"_plot.pdf", fig)
