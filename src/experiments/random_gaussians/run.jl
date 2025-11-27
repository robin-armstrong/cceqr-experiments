include("../../config/config.jl")

using LinearAlgebra
using CairoMakie
using StatsBase
using Random
using CCEQR
using JLD2

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

rng       = MersenneTwister(1)
m_range   = round.(Int64, exp10.(range(1, 3, 15)))
m_fixed   = 50
n_range   = round.(Int64, exp10.(range(2, 6, 15)))
n_fixed   = 1000
rho_range = exp10.(range(-5, -.3, 20))
rho_fixed = 1e-2
numtrials = 10
readme    = "Comparing GEQP3 and CCEQR on Gaussian matrices of various sizes."

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

    function run_gaussian_experiment(rng, m_range, m_fixed, n_range, n_fixed,
                                     rho_range, rho_fixed, numtrials, readme,
                                     destination)
        # recording information about this experiment

        logstr  = "rng       = "*string(rng)*"\n"
        logstr *= "m_range   = "*string(m_range)*"\n"
        logstr *= "m_fixed   = "*string(m_fixed)*"\n"
        logstr *= "n_range   = "*string(n_range)*"\n"
        logstr *= "n_fixed   = "*string(n_fixed)*"\n"
        logstr *= "rho_range = "*string(rho_range)*"\n"
        logstr *= "rho_fixed = "*string(rho_fixed)*"\n"
        logstr *= "numtrials = "*string(numtrials)*"\n"
        logstr *= "\n"*readme*"\n"

        logfile = destination*"_log.txt"
        touch(logfile)
        io = open(logfile, "w")
        write(io, logstr)
        close(io)

        fprintln(logstr)

        data_m   = Dict()
        data_n   = Dict()
        data_rho = Dict()

        for alg in ["cceqr_cssp", "cceqr_cpqr", "geqp3"]
            data_m[alg]   = zeros(length(m_range), numtrials)
            data_n[alg]   = zeros(length(n_range), numtrials)
            data_rho[alg] = zeros(length(rho_range), numtrials)
        end
        
        total_trials  = numtrials*(length(m_range)+
                                   length(n_range)+
                                   length(rho_range))
        trial_counter = 0

        for (m_idx, m) in enumerate(m_range)
            A   = randn(rng, m, n_fixed)
            tmp = zeros(m, n_fixed)

            fprintln("\nSHAPE "*string(size(A))*", "*
                       "RHO = "*string(rho_fixed))
            fprintln("--------------------")

            copy!(tmp, A)
            p_geqp3 = qr!(tmp, ColumnNorm()).p

            copy!(tmp, A)
            cceqr!(tmp, rho = rho_fixed)

            copy!(tmp, A)
            p_cceqr, _, _, _ = cceqr!(tmp, rho = rho_fixed, full = true)

            if p_geqp3[1:min(m, n_fixed)] != p_cceqr[1:min(m, n_fixed)]
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                @save destination*"_failure_data.jld2" A rho_fixed j expected got
                throw(error("incorrect permutation from cceqr"))
            end

            for trial_index = 1:numtrials
                trial_counter += 1
                fprintln("    trial "*string(trial_counter)*
                         " of "*string(total_trials)*"...")

                copy!(tmp, A)
                data_m["geqp3"][m_idx, trial_index] = @elapsed qr!(tmp, ColumnNorm())

                copy!(tmp, A)
                data_m["cceqr_cssp"][m_idx, trial_index] = @elapsed cceqr!(tmp, rho = rho_fixed)

                copy!(tmp, A)
                data_m["cceqr_cpqr"][m_idx, trial_index] = @elapsed cceqr!(tmp, rho = rho_fixed, full = true)
            end

            @save destination*"_data.jld2" m_range m_fixed n_range n_fixed rho_range rho_fixed numtrials data_m data_n data_rho
        end

        for (n_idx, n) in enumerate(n_range)
            A   = randn(rng, m_fixed, n)
            tmp = zeros(m_fixed, n)

            fprintln("\nSHAPE "*string(size(A))*", "*
                       "RHO = "*string(rho_fixed))
            fprintln("--------------------")

            copy!(tmp, A)
            p_geqp3 = qr!(tmp, ColumnNorm()).p

            copy!(tmp, A)
            cceqr!(tmp, rho = rho_fixed)

            copy!(tmp, A)
            p_cceqr, _, _, _ = cceqr!(tmp, rho = rho_fixed, full = true)

            if p_geqp3[1:min(m_fixed, n)] != p_cceqr[1:min(m_fixed, n)]
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                @save destination*"_failure_data.jld2" A rho_fixed j expected got
                throw(error("incorrect permutation from cceqr"))
            end

            for trial_index = 1:numtrials
                trial_counter += 1
                fprintln("    trial "*string(trial_counter)*
                         " of "*string(total_trials)*"...")

                copy!(tmp, A)
                data_n["geqp3"][n_idx, trial_index] = @elapsed qr!(tmp, ColumnNorm())

                copy!(tmp, A)
                data_n["cceqr_cssp"][n_idx, trial_index] = @elapsed cceqr!(tmp, rho = rho_fixed)

                copy!(tmp, A)
                data_n["cceqr_cpqr"][n_idx, trial_index] = @elapsed cceqr!(tmp, rho = rho_fixed, full = true)
            end

            @save destination*"_data.jld2" m_range m_fixed n_range n_fixed rho_range rho_fixed numtrials data_m data_n data_rho
        end

        for (rho_idx, rho) in enumerate(rho_range)
            A   = randn(rng, m_fixed, n_fixed)
            tmp = zeros(m_fixed, n_fixed)

            fprintln("\nSHAPE "*string(size(A))*" ,"*
                       "RHO = "*string(rho))
            fprintln("--------------------")

            copy!(tmp, A)
            p_geqp3 = qr!(tmp, ColumnNorm()).p

            copy!(tmp, A)
            cceqr!(tmp, rho = rho)

            copy!(tmp, A)
            p_cceqr, _, _, _ = cceqr!(tmp, rho = rho, full = true)
            p_cceqr          = p_cceqr[1:n_fixed]

            if p_geqp3[1:min(m_fixed, n_fixed)] != p_cceqr[1:min(m_fixed, n_fixed)]
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                @save destination*"_failure_data.jld2" A rho j expected got
                throw(error("incorrect permutation from cceqr"))
            end

            for trial_index = 1:numtrials
                trial_counter += 1
                fprintln("    trial "*string(trial_counter)*
                         " of "*string(total_trials)*"...")

                copy!(tmp, A)
                data_rho["geqp3"][rho_idx, trial_index] = @elapsed qr!(tmp, ColumnNorm())

                copy!(tmp, A)
                data_rho["cceqr_cssp"][rho_idx, trial_index] = @elapsed cceqr!(tmp, rho = rho)

                copy!(tmp, A)
                data_rho["cceqr_cpqr"][rho_idx, trial_index] = @elapsed cceqr!(tmp, rho = rho, full = true)
            end

            @save destination*"_data.jld2" m_range m_fixed n_range n_fixed rho_range rho_fixed numtrials data_m data_n data_rho
        end
    end

    run_gaussian_experiment(rng, m_range, m_fixed, n_range, n_fixed,
                            rho_range, rho_fixed, numtrials, readme,
                            destination)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" m_range m_fixed n_range n_fixed rho_range rho_fixed numtrials data_m data_n data_rho

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (500, 1000), fonts = (; regular = regfont))

m_plot = Axis(fig[1,1],
              xlabel             = "Number of Rows ("*string(n_fixed)*" Columns, ρ = "*string(rho_fixed)*")",
              xscale             = log10,
              xminorticksvisible = true,
              xminorgridvisible  = true,
              xminorticks        = IntervalsBetween(10),
              ylabel             = "Runtime (s)",
              yscale             = log10,
              yminorticksvisible = true,
              yminorgridvisible  = true,
              yminorticks        = IntervalsBetween(10))

cceqr_cssp_median = vec(median(data_m["cceqr_cssp"], dims = 2))
cceqr_cpqr_median = vec(median(data_m["cceqr_cpqr"], dims = 2))
geqp3_median      = vec(median(data_m["geqp3"], dims = 2))

scatterlines!(m_plot, m_range, cceqr_cssp_median, color = :blue, marker = :diamond, label = "CCEQR (CSSP Only)")
scatterlines!(m_plot, m_range, cceqr_cpqr_median, color = :transparent, strokecolor = :green, strokewidth = 2, linewidth = 2, marker = :circle, markersize = 15, label = "CCEQR (full CPQR)")
lines!(m_plot, m_range, cceqr_cpqr_median, color = :green)
lines!(m_plot, m_range, geqp3_median, color = :red, linestyle = :dash, label = "GEQP3")
lines!(m_plot, m_range, 5e-7*(m_range).^2, color = :black, linestyle = :dashdot, label = L"\mathcal{O}(m^2)\text{ (reference)}")

axislegend(m_plot, position = :lt)

n_plot = Axis(fig[2,1],
              xlabel             = "Number of Columns ("*string(m_fixed)*" Rows, ρ = "*string(rho_fixed)*")",
              xscale             = log10,
              xminorticksvisible = true,
              xminorgridvisible  = true,
              xminorticks        = IntervalsBetween(10),
              ylabel             = "Runtime (s)",
              yscale             = log10,
              yminorticksvisible = true,
              yminorgridvisible  = true,
              yminorticks        = IntervalsBetween(10))

cceqr_cssp_median = vec(median(data_n["cceqr_cssp"], dims = 2))
cceqr_cpqr_median = vec(median(data_n["cceqr_cpqr"], dims = 2))
geqp3_median      = vec(median(data_n["geqp3"], dims = 2))

scatterlines!(n_plot, n_range, cceqr_cssp_median, color = :blue, marker = :diamond, label = "CCEQR (CSSP Only)")
scatterlines!(n_plot, n_range, cceqr_cpqr_median, color = :transparent, strokecolor = :green, strokewidth = 2, linewidth = 2, marker = :circle, markersize = 15, label = "CCEQR (full CPQR)")
lines!(n_plot, n_range, cceqr_cpqr_median, color = :green)
lines!(n_plot, n_range, geqp3_median, color = :red, linestyle = :dash, label = "GEQP3")
lines!(n_plot, n_range, 5e-7*n_range, color = :black, linestyle = :dashdot, label = L"\mathcal{O}(n)\text{ (reference)}")

axislegend(n_plot, position = :lt)

rho_plot = Axis(fig[3,1],
                xlabel             = "ρ ("*string(m_fixed)*" Rows, "*string(n_fixed)*" Columns)",
                xscale             = log10,
                xminorticksvisible = true,
                xminorgridvisible  = true,
                xminorticks        = IntervalsBetween(10),
                ylabel             = "Runtime (s)",
                yminorticksvisible = true,
                yminorgridvisible  = true,
                yminorticks        = IntervalsBetween(10))

cceqr_cssp_median = vec(median(data_rho["cceqr_cssp"], dims = 2))
cceqr_cpqr_median = vec(median(data_rho["cceqr_cpqr"], dims = 2))
geqp3_median      = median(data_rho["geqp3"])

scatterlines!(rho_plot, rho_range, cceqr_cssp_median, color = :blue, marker = :diamond, label = "CCEQR (CSSP Only)")
scatterlines!(rho_plot, rho_range, cceqr_cpqr_median, color = :transparent, strokecolor = :green, strokewidth = 2, linewidth = 2, marker = :circle, markersize = 15, label = "CCEQR (full CPQR)")
lines!(rho_plot, rho_range, cceqr_cpqr_median, color = :green)
hlines!(rho_plot, geqp3_median, color = :red, linestyle = :dash, label = "GEQP3")

axislegend(rho_plot, position = :lb)

save(destination*"_plot.pdf", fig)
