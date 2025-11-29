include("../../config/config.jl")

using LinearAlgebra
using CairoMakie
using StatsBase
using CCEQR
using JLD2

include("../../misc/fprints.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

molecule  = "alkane"     # either "water" or "alkane"

rho_range = exp10.(range(-5, -.3, 20))
rho_fixed = 1e-2
th_range  = 1:16
th_fixed  = 8
numtrials = 30

plot_only   = false      # if "true" then data will be read from disk and not regenerated
destination = "src/experiments/wannier_localization/alkane"
readme      = "Comparing GEQP3 and CCEQR on the alkane example."

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

function run_wannier_experiment(molecule, rho_range, rho_fixed, th_range,
                                th_fixed, numtrials, destination, readme)
    logstr  = "molecule  = "*molecule*"\n"
    logstr *= "rho_range = "*string(rho_range)*"\n"
    logstr *= "rho_fixed = "*string(rho_fixed)*"\n"
    logstr *= "th_range   = "*string(th_range)*"\n"
    logstr *= "th_fixed   = "*string(th_fixed)*"\n"
    logstr *= "numtrials = "*string(numtrials)*"\n"
    logstr *= "\n"*readme*"\n"

    logfile = destination*"_log.txt"
    touch(logfile)
    io = open(logfile, "w")
    write(io, logstr)
    close(io)

    fprintln(logstr)

    cceqr_cssp_vsrho = zeros(length(rho_range), numtrials)
    cceqr_cpqr_vsrho = zeros(length(rho_range), numtrials)
    geqp3_vsrho      = zeros(numtrials)

    cceqr_cssp_vsth = zeros(length(th_range), numtrials)
    cceqr_cpqr_vsth = zeros(length(th_range), numtrials)
    geqp3_vsth      = zeros(length(th_range), numtrials)
    
    cceqr_cycles = zeros(length(rho_range))
    cceqr_avgblk = zeros(length(rho_range))
    cceqr_active = zeros(length(rho_range))

    totaltrials = (length(rho_range) + length(th_range))*numtrials
    trialcount  = 0

    fprintln("loading Wannier basis...")
    @load "data/matrices/"*molecule*".jld2" Psi Ls Ns
    
    m, n = size(Psi)
    Vt   = zeros(m, n)

    BLAS.set_num_threads(th_fixed)
    fprintln("testing GEQP3 with "*string(th_fixed)*" threads...")

    copy!(Vt, Psi)
    p_geqp3 = qr!(Vt, ColumnNorm()).p

    for trial_idx = 1:numtrials
        copy!(Vt, Psi)
        geqp3_vsrho[trial_idx] = @elapsed qr!(Vt, ColumnNorm())
    end

    fprintln("testing CCEQR with "*string(th_fixed)*" threads over rho ...")

    for (rho_idx, rho) in enumerate(rho_range)
        copy!(Vt, Psi)
        cceqr!(Vt, rho = rho)

        copy!(Vt, Psi)
        p_cceqr, blocks, avg_b, act = cceqr!(Vt, rho = rho, full = true)
        p_cceqr                     = p_cceqr[1:m]

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

        for trial_idx = 1:numtrials
            trialcount += 1
            fprintln("    trial "*string(trialcount)*" of "*string(totaltrials)*"...")

            copy!(Vt, Psi)
            t = @elapsed cceqr!(Vt, rho = rho)
            cceqr_cssp_vsrho[rho_idx, trial_idx] = t

            copy!(Vt, Psi)
            t = @elapsed cceqr!(Vt, rho = rho, full = true)
            cceqr_cpqr_vsrho[rho_idx, trial_idx] = t
        end

        @save destination*"_data.jld2" molecule rho_range rho_fixed th_range th_fixed numtrials geqp3_vsrho cceqr_cssp_vsrho cceqr_cpqr_vsrho cceqr_cssp_vsth cceqr_cpqr_vsth geqp3_vsth cceqr_cycles cceqr_avgblk cceqr_active
    end

    fprintln("testing GEQP3 and CCEQR over thread numbers with rho = "*string(rho_fixed)*"...")

    for (th_idx, th) in enumerate(th_range)
        BLAS.set_num_threads(th)

        copy!(Vt, Psi)
        qr!(Vt, ColumnNorm())
        
        copy!(Vt, Psi)
        cceqr!(Vt, rho = rho_fixed)

        copy!(Vt, Psi)
        cceqr!(Vt, rho = rho_fixed, full = true)

        for trial_idx = 1:numtrials
            trialcount += 1
            fprintln("    trial "*string(trialcount)*" of "*string(totaltrials)*"...")

            copy!(Vt, Psi)
            t = @elapsed qr!(Vt, ColumnNorm())
            geqp3_vsth[th_idx, trial_idx] = t
            
            copy!(Vt, Psi)
            t = @elapsed cceqr!(Vt, rho = rho_fixed)
            cceqr_cssp_vsth[th_idx, trial_idx] = t

            copy!(Vt, Psi)
            t = @elapsed cceqr!(Vt, rho = rho_fixed, full = true)
            cceqr_cpqr_vsth[th_idx, trial_idx] = t

            @save destination*"_data.jld2" molecule rho_range rho_fixed th_range th_fixed numtrials geqp3_vsrho cceqr_cssp_vsrho cceqr_cpqr_vsrho cceqr_cssp_vsth cceqr_cpqr_vsth geqp3_vsth cceqr_cycles cceqr_avgblk cceqr_active
        end
    end
end

plot_only || run_wannier_experiment(molecule, rho_range, rho_fixed, th_range,
                                    th_fixed, numtrials, destination, readme)

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" molecule rho_range rho_fixed th_range th_fixed numtrials geqp3_vsrho cceqr_cssp_vsrho cceqr_cpqr_vsrho cceqr_cssp_vsth cceqr_cpqr_vsth geqp3_vsth cceqr_cycles cceqr_avgblk cceqr_active

cceqr_median_cssp  = vec(median(cceqr_cssp_vsrho, dims = 2))
cceqr_median_cpqr  = vec(median(cceqr_cpqr_vsrho, dims = 2))
geqp3_median_times = median(geqp3_vsrho)
tmin               = .8*min(geqp3_median_times, minimum(cceqr_median_cssp), minimum(cceqr_median_cpqr))
tmax               = 1.5*max(geqp3_median_times, maximum(cceqr_median_cssp), maximum(cceqr_median_cpqr))

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (650, 650), fonts = (; regular = regfont))

time = Axis(fig[1,1],
            limits             = (nothing, nothing, tmin, tmax),
            xlabel             = "ρ",
            xminorticksvisible = true,
            xminorgridvisible  = true,
            xminorticks        = IntervalsBetween(10),
            xscale             = log10,
            ylabel             = "Runtime (s)",
            yminorticksvisible = true,
            yminorgridvisible  = true,
            yminorticks        = IntervalsBetween(10)
           )

scatterlines!(time, rho_range, cceqr_median_cssp, color = :blue, marker = :diamond, label = "CCEQR (CSSP only)")
scatterlines!(time, rho_range, cceqr_median_cpqr, color = :transparent, strokecolor = :green, strokewidth = 2, linewidth = 2, marker = :circle, markersize = 15, label = "CCEQR (full CPQR)")
lines!(time, rho_range, cceqr_median_cpqr, color = :green)
hlines!(time, geqp3_median_times, color = :red, linestyle = :dash, label = "GEQP3")
axislegend(time, position = :lt)

cycles = Axis(fig[1,2],
              xlabel             = "ρ",
              xminorticksvisible = true,
              xminorgridvisible  = true,
              xminorticks        = IntervalsBetween(10),
              xscale             = log10,
              ylabel             = "CCEQR Cycle Count",
              yminorticksvisible = true,
              yminorgridvisible  = true,
              yminorticks        = IntervalsBetween(10))
             
lines!(cycles, rho_range, cceqr_cycles, color = :black)

cceqr_median_cssp  = vec(median(cceqr_cssp_vsth, dims = 2))
cceqr_median_cpqr  = vec(median(cceqr_cpqr_vsth, dims = 2))
geqp3_median_times = vec(median(geqp3_vsth, dims = 2))

threads = Axis(fig[2,1:2],
               width              = Relative(0.5),
               xlabel             = "BLAS/LAPACK Threads",
               xticks             = 2:2:maximum(th_range),
               ylabel             = "Runtime (s)",
               yminorticksvisible = true,
               yminorgridvisible  = true,
               yminorticks        = IntervalsBetween(10))

scatterlines!(threads, th_range, cceqr_median_cssp, color = :blue, marker = :diamond, label = "CCEQR (CSSP Only)")
scatterlines!(threads, th_range, cceqr_median_cpqr, color = :transparent, strokecolor = :green, strokewidth = 2, linewidth = 2, marker = :circle, markersize = 15, label = "CCEQR (full CPQR)")
lines!(threads, th_range, cceqr_median_cpqr, color = :green)
lines!(threads, th_range, geqp3_median_times, color = :red, linestyle = :dash, label = "GEQP3")

axislegend(threads)

save(destination*"_plot.pdf", fig)
