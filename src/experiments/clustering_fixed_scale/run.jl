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

nrange = round.(Int64, exp10.(range(2, 6, 10)))     # number of data points, must be > 4*k
k      = 20                                         # number of Gaussian mixture components
scale  = 6.
noise  = 1.

kernel    = "gaussian"  # type of kernel function      
bandwidth = 10.         # bandwidth of kernel function

rho_range = exp10.(range(-5, -.3, 20))
numtrials = 20

plot_only   = false     # if "true" then data will be read from disk and not regenerated
destination = "src/experiments/clustering_fixed_scale/cluster"
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
        geqp3_time      = zeros(numtrials)

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

    run_cluster_fixed_scale(rng, nrange, k, scale, noise, kernel, bandwidth,
                            rho_range, numtrials, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" nrange k scale rho_range skill_learned skill_random numtrials geqp3_time cceqr_time_cssp cceqr_time_cpqr cceqr_cycles cceqr_avgblk cceqr_active

cceqr_mean_cssp = mean(cceqr_time_cssp, dims = 3)
cceqr_mean_cpqr = mean(cceqr_time_cpqr, dims = 3)
geqp3_mean      = mean(geqp3_time)

cceqr_mean_cssp = reshape(cceqr_mean_cssp, (length(nrange), length(rho_range)))
cceqr_mean_cpqr = reshape(cceqr_mean_cpqr, (length(nrange), length(rho_range)))

time_comp_cssp = geqp3_mean*cceqr_mean_cssp.^(-1)
time_comp_cpqr = geqp3_mean*cceqr_mean_cpqr.^(-1)

extremes = extrema(log10.([time_comp_cssp; time_comp_cpqr]))

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (900, 400))

time_cssp = Axis(fig[1,1],
                 title  = L"$\log_{10}(T_\mathrm{GEQP3}/T_\mathrm{CCEQR})$ (CSSP Only)",
                 xlabel = L"$\log_{10} \,\rho$",
                 ylabel = L"Dataset Size ($\log_{10} n$)",
                )
heatmap!(time_cssp, log10.(rho_range), log10.(nrange), transpose(log10.(time_comp_cssp)), colormap = :vik, colorrange = extremes)

time_cpqr = Axis(fig[1,2],
                 title  = L"$\log_{10}(T_\mathrm{GEQP3}/T_\mathrm{CCEQR})$ (Full CPQR)",
                 xlabel = L"$\log_{10} \,\rho$",
                 ylabel = L"Dataset Size ($\log_{10} n$)",
                )
heatmap!(time_cpqr, log10.(rho_range), log10.(nrange), transpose(log10.(time_comp_cpqr)), colormap = :vik, colorrange = extremes)
Colorbar(fig[1,3], colormap = :vik, limits = extremes)

save(destination*"_plot.pdf", fig)
