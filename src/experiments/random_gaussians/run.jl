using LinearAlgebra
using CairoMakie
using StatsBase
using Random
using CCEQR
using JLD2

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng         = MersenneTwister(1)
k           = 50
n_range     = round.(Int64, exp10.(range(2, 6, 15)))
rho         = 1e-2
numtrials   = 10
readme      = "Comparing GEQP3 and CCEQR on Gaussian random test matrices of various sizes."

plot_only   = false
destination = "src/experiments/random_gaussians/gaussian_test"

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

if !plot_only
    function fprintln(s)
        println(s)
        flush(stdout)
    end

    function run_gaussian_experiment(rng, k, n_range, rho, numtrials, readme, destination)
        # recording information about this experiment

        logstr  = "rng         = "*string(rng)*"\n"
        logstr *= "k           = "*string(k)*"\n"
        logstr *= "n_range     = "*string(n_range)*"\n"
        logstr *= "rho         = "*string(rho)*"\n"
        logstr *= "numtrials   = "*string(numtrials)*"\n"
        logstr *= "\n"*readme*"\n"

        logfile = destination*"_log.txt"
        touch(logfile)
        io = open(logfile, "w")
        write(io, logstr)
        close(io)

        fprintln(logstr)

        cceqr_runtimes = zeros(length(n_range), numtrials)
        geqp3_runtimes = zeros(length(n_range), numtrials)
        total_trials   = length(n_range)*numtrials
        trial_counter  = 0
        
        for n_index = 1:length(n_range)
            n   = n_range[n_index]
            A   = randn(rng, k, n)
            tmp = zeros(k, n)

            fprintln("\nSHAPE "*string(n_index)*" OF "*string(length(n_range)))
            fprintln("--------------------")

            copy!(tmp, A)
            p_geqp3 = qr!(tmp, ColumnNorm()).p

            copy!(tmp, A)
            p_cceqr, _, _, _ = cceqr!(tmp, rho = rho)

            if(p_geqp3[1:k] != p_cceqr)
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                @save destination*"_failure_data.jld2" A rho j expected got
                throw(error("incorrect permutation from cceqr"))
            end

            for trial_index = 1:numtrials
                trial_counter += 1
                fprintln("    trial "*string(trial_counter)*" of "*string(total_trials)*"...")

                copy!(tmp, A)
                geqp3_runtimes[n_index, trial_index] = @elapsed qr!(tmp, ColumnNorm())

                copy!(tmp, A)
                cceqr_runtimes[n_index, trial_index] = @elapsed cceqr!(tmp, rho = rho)
            end

            @save destination*"_data.jld2" k n_range geqp3_runtimes cceqr_runtimes
        end
    end

    run_gaussian_experiment(rng, k, n_range, rho, numtrials, readme, destination)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" k n_range geqp3_runtimes cceqr_runtimes

geqp3_mean_times = vec(mean(geqp3_runtimes, dims = 2))
cceqr_mean_times = vec(mean(cceqr_runtimes, dims = 2))

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (500, 300))

runtimes = Axis(fig[1,1],
                title  = "Runtimes on Gaussian Matrices",
                xlabel = "Number of Columns ("*string(k)*" Rows)",
                ylabel = "Runtime (s)",
                xscale = log10,
                yscale = log10)

lines!(runtimes, n_range, geqp3_mean_times, color = :red, label = "GEQP3")
lines!(runtimes, n_range, cceqr_mean_times, color = :blue, label = "CCEQR")
axislegend(runtimes, position = :lt)
save(destination*"_plot.pdf", fig)
