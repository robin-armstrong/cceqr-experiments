using LinearAlgebra
using CairoMakie
using StatsBase
using CCEQR
using JLD2

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

molecule  = "alkane"     # either "water" or "alkane"

rho_range = exp10.(range(-5, -.3, 20))
numtrials = 10

plot_only   = false      # if "true" then data will be read from disk and not regenerated
destination = "src/experiments/wannier_localization/alkane"
readme      = "Comparing GEQP3 and CCEQR on the alkane example."

##########################################################################
######################## DATA GENERATION #################################
##########################################################################
if !plot_only
    function fprintln(s)
        println(s)
        flush(stdout)
    end

    function run_wannier_experiment(molecule, rho_range, numtrials, destination, readme)
        logstr  = "molecule  = "*molecule*"\n"
        logstr *= "rho_range = "*string(rho_range)*"\n"
        logstr *= "numtrials = "*string(numtrials)*"\n"
        logstr *= "\n"*readme*"\n"

        logfile = destination*"_log.txt"
        touch(logfile)
        io = open(logfile, "w")
        write(io, logstr)
        close(io)

        fprintln(logstr)

        cceqr_time_cssp = zeros(length(rho_range), numtrials)
        cceqr_time_cpqr = zeros(length(rho_range), numtrials)
        cceqr_cycles    = zeros(length(rho_range))
        cceqr_avgblk    = zeros(length(rho_range))
        cceqr_active    = zeros(length(rho_range))
        geqp3_time      = zeros(numtrials)

        totaltrials = length(rho_range)*numtrials
        trialcount  = 0

        fprintln("loading Wannier basis...")
        @load "data/matrices/"*molecule*".jld2" Psi Ls Ns
        
        m, n = size(Psi)
        Vt   = zeros(m, n)

        fprintln("testing GEQP3...")

        copy!(Vt, Psi)
        p_geqp3 = qr!(Vt, ColumnNorm()).p

        for i = 1:numtrials
            copy!(Vt, Psi)
            geqp3_time[i] = @elapsed qr!(Vt, ColumnNorm())
        end

        fprintln("testing CCEQR...")

        for (rho_idx, rho) in enumerate(rho_range)
            copy!(Vt, Psi)
            p_cceqr, blocks, avg_b, act = cceqr!(Vt, rho = rho)

            copy!(Vt, Psi)
            p_cceqr, blocks, avg_b, act = cceqr!(Vt, rho = rho, full = true)

            if(p_geqp3[1:m] != p_cceqr)
                j = 1
                while(p_geqp3[j] == p_cceqr[j]) j += 1 end

                expected = p_geqp3[j]
                got      = p_cceqr[j]

                copy!(Vt, Psi)
                @save destination*"_failure_data.jld2" Vt rho j expected got
                throw(error("incorrect permutation from cceqr"))
            end

            cceqr_cycles[rho_idx] = blocks
            cceqr_avgblk[rho_idx] = avg_b/n
            cceqr_active[rho_idx] = act/n

            for trial_index = 1:numtrials
                trialcount += 1
                fprintln("    trial "*string(trialcount)*" of "*string(totaltrials)*"...")

                copy!(Vt, Psi)
                t = @elapsed cceqr!(Vt, rho = rho)
                cceqr_time_cssp[rho_idx, trial_index] = t

                copy!(Vt, Psi)
                t = @elapsed cceqr!(Vt, rho = rho, full = true)
                cceqr_time_cpqr[rho_idx, trial_index] = t
            end

            @save destination*"_data.jld2" molecule rho_range numtrials geqp3_time cceqr_time_cssp cceqr_time_cpqr cceqr_cycles cceqr_avgblk cceqr_active
        end
    end

    run_wannier_experiment(molecule, rho_range, numtrials, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" molecule rho_range numtrials geqp3_time cceqr_time_cssp cceqr_time_cpqr cceqr_cycles cceqr_avgblk cceqr_active

cceqr_median_cssp  = vec(median(cceqr_time_cssp, dims = 2))
cceqr_median_cpqr  = vec(median(cceqr_time_cpqr, dims = 2))
geqp3_median_times = median(geqp3_time)
tmin               = .8*min(geqp3_median_times, minimum(cceqr_median_cssp), minimum(cceqr_median_cpqr))
tmax               = 1.5*max(geqp3_median_times, maximum(cceqr_median_cssp), maximum(cceqr_median_cpqr))

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (650, 325))

time = Axis(fig[1,1],
            limits             = (nothing, nothing, tmin, tmax),
            xlabel             = L"$\rho$",
            xminorticksvisible = true,
            xminorgridvisible  = true,
            xminorticks        = IntervalsBetween(10),
            xscale             = log10,
            ylabel             = L"\text{Runtime (s)}",
            yminorticksvisible = true,
            yminorgridvisible  = true,
            yminorticks        = IntervalsBetween(10)
           )

scatterlines!(time, rho_range, cceqr_median_cssp, color = :blue, marker = :diamond, label = L"\text{CCEQR (CSSP only)}")
scatterlines!(time, rho_range, cceqr_median_cpqr, color = :green, marker = :circle, label = L"\text{CCEQR (full CPQR)}")
hlines!(time, geqp3_median_times, color = :red, linestyle = :dash, label = L"\text{GEQP3}")
axislegend(time, position = :lt)

cycles = Axis(fig[1,2],
              xlabel             = L"$\rho$",
              xminorticksvisible = true,
              xminorgridvisible  = true,
              xminorticks        = IntervalsBetween(10),
              xscale             = log10,
              ylabel             = L"\text{CCEQR Cycle Count}",
              yminorticksvisible = true,
              yminorgridvisible  = true,
              yminorticks        = IntervalsBetween(10)
             )
lines!(cycles, rho_range, cceqr_cycles, color = :black)

save(destination*"_plot.pdf", fig)
