using LinearAlgebra
using CairoMakie
using StatsBase
using Hadamard
using CCEQR
using JLD2

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

k         = 32                                       # must be a multiple of 2
n_range   = round.(Int64, exp2.(range(6, 20, 15)))   # must be multiples of 2
rho       = 1e-2
numtrials = 10
readme    = "Comparing GEQP3 and CCEQR on Hadamard test matrices of various sizes."

plot_only   = false
destination = "src/experiments/hadamard/hadamard_test"

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

if !plot_only
    function fprintln(s)
        println(s)
        flush(stdout)
    end

    function run_hadamard_experiment(k, n_range, rho, numtrials, readme, destination)
        # recording information about this experiment

        logstr  = "k           = "*string(k)*"\n"
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

        cceqr_cssp_times = zeros(length(n_range), numtrials)
        cceqr_cpqr_times = zeros(length(n_range), numtrials)
        geqp3_runtimes   = zeros(length(n_range), numtrials)
        total_trials     = length(n_range)*numtrials
        trial_counter    = 0
        
        for n_index = 1:length(n_range)
            n   = n_range[n_index]
            d   = div(n, k)
            tmp = zeros(k, n)
            A   = zeros(k, n)
            H   = hadamard(k)
            
            for i = 1:k
                idx_A      = ((i-1)*d + 1):(i*d)
                A[:,idx_A] = H[:, i]*ones(1, d)
            end
            
            D = Diagonal(reverse(1 .+ 1000*eps()*(1:n)))
            rmul!(A, D)

            fprintln("\nSHAPE "*string(n_index)*" OF "*string(length(n_range)))
            fprintln("--------------------")

            copy!(tmp, A)
            p_geqp3 = qr!(tmp, ColumnNorm()).p

            copy!(tmp, A)
            cceqr!(tmp, rho = rho)

            copy!(tmp, A)
            p_cceqr, _, _, _ = cceqr!(tmp, rho = rho, full = true)
            p_cceqr          = p_cceqr[1:k]

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
                cceqr_cssp_times[n_index, trial_index] = @elapsed cceqr!(tmp, rho = rho)

                copy!(tmp, A)
                cceqr_cpqr_times[n_index, trial_index] = @elapsed cceqr!(tmp, rho = rho, full = true)
            end

            @save destination*"_data.jld2" k n_range geqp3_runtimes cceqr_cssp_times cceqr_cpqr_times
        end
    end

    A = run_hadamard_experiment(k, n_range, rho, numtrials, readme, destination)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" k n_range geqp3_runtimes cceqr_cssp_times cceqr_cpqr_times

cceqr_cssp_median = vec(median(cceqr_cssp_times, dims = 2))
cceqr_cpqr_median = vec(median(cceqr_cpqr_times, dims = 2))
geqp3_median      = vec(median(geqp3_runtimes, dims = 2))

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (500, 300))

runtimes = Axis(fig[1,1],
                xlabel             = L"\text{Number of Columns (32 Rows)}",
                xminorticksvisible = true,
                xminorgridvisible  = true,
                xminorticks        = IntervalsBetween(10),
                xscale             = log10,
                ylabel             = L"\text{Runtime (s)}",
                yminorticksvisible = true,
                yminorgridvisible  = true,
                yminorticks        = IntervalsBetween(10),
                yscale             = log10)

scatterlines!(runtimes, n_range, cceqr_cssp_median, color = :blue, marker = :diamond, label = L"\text{CCEQR (CSSP Only)}")
scatterlines!(runtimes, n_range, cceqr_cpqr_median, color = :green, marker = :circle, label = L"\text{CCEQR (full CPQR)}")
lines!(runtimes, n_range, geqp3_median, color = :red, linestyle = :dash, label = L"\text{GEQP3}")
lines!(runtimes, n_range, 5e-7*n_range, color = :black, linestyle = :dashdot, label = L"\mathcal{O}(n)\text{ (reference)}")
axislegend(runtimes, position = :lt)
save(destination*"_plot.pdf", fig)
