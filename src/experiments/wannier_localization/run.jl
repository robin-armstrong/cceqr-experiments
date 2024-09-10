using LinearAlgebra
using CairoMakie
using StatsBase
using JLD2

include("../../algorithms/cceqr.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

molecule  = "alkane"     # either "water" or "alkane"

rho_range = exp10.(range(-5, -.3, 20))
eta_range = exp10.(range(-5, -.3, 20))
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

    function run_wannier_experiment(molecule, rho_range, eta_range, numtrials, destination, readme)
        logstr  = "molecule  = "*molecule*"\n"
        logstr *= "rho_range = "*string(rho_range)*"\n"
        logstr *= "eta_range = "*string(eta_range)*"\n"
        logstr *= "numtrials = "*string(numtrials)*"\n"
        logstr *= "\n"*readme*"\n"

        logfile = destination*"_log.txt"
        touch(logfile)
        io = open(logfile, "w")
        write(io, logstr)
        close(io)

        fprintln(logstr)

        cceqr_time    = zeros(length(rho_range), length(eta_range), numtrials)
        cceqr_cycles  = zeros(length(rho_range), length(eta_range))
        cceqr_avgblk  = zeros(length(rho_range), length(eta_range))
        cceqr_active  = zeros(length(rho_range), length(eta_range))
        geqp3_time    = zeros(numtrials)

        totaltrials = length(rho_range)*length(eta_range)*numtrials
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

        for rho_index = 1:length(rho_range)
            rho = rho_range[rho_index]

            for eta_index = 1:length(eta_range)
                eta = eta_range[eta_index]

                copy!(Vt, Psi)
                p_cceqr, blocks, avg_b, act = cceqr!(Vt, eta = eta, rho = rho)

                if(p_geqp3[1:m] != p_cceqr)
                    j = 1
                    while(p_geqp3[j] == p_cceqr[j]) j += 1 end
    
                    expected = p_geqp3[j]
                    got      = p_cceqr[j]
    
                    copy!(Vt, Psi)
                    @save destination*"_failure_data.jld2" Vt rho eta j expected got
                    throw(error("incorrect permutation from cceqr"))
                end

                cceqr_cycles[rho_index, eta_index] = blocks
                cceqr_avgblk[rho_index, eta_index] = avg_b/n
                cceqr_active[rho_index, eta_index] = act/n

                for trial_index = 1:numtrials
                    trialcount += 1
                    fprintln("    trial "*string(trialcount)*" of "*string(totaltrials)*"...")

                    copy!(Vt, Psi)
                    t = @elapsed cceqr!(Vt, eta = eta, rho = rho)
                    cceqr_time[rho_index, eta_index, trial_index] = t
                end

                @save destination*"_data.jld2" molecule rho_range eta_range numtrials geqp3_time cceqr_time cceqr_cycles cceqr_avgblk cceqr_active
            end
        end
    end

    run_wannier_experiment(molecule, rho_range, eta_range, numtrials, destination, readme)
end

##########################################################################
######################## PLOTTING ########################################
##########################################################################

@load destination*"_data.jld2" molecule rho_range eta_range numtrials geqp3_time cceqr_time cceqr_cycles cceqr_avgblk cceqr_active

cceqr_mean_times = zeros(length(rho_range), length(eta_range))

for i = 1:length(rho_range)
    for j = 1:length(eta_range)
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
            ylabel = L"$\log_{10} \,\eta$"
           )
heatmap!(time, log10.(rho_range), log10.(eta_range), time_comp)
Colorbar(fig[1,2], limits = extrema(time_comp))

blocks = Axis(fig[1,3],
              title  = "Average Block Percentage Per Cycle",
              xlabel = L"$\log_{10} \,\rho$",
              ylabel = L"$\log_{10} \,\eta$"
             )
heatmap!(blocks, log10.(rho_range), log10.(eta_range), cceqr_avgblk)
Colorbar(fig[1,4], limits = extrema(cceqr_avgblk))

active = Axis(fig[2,1],
              title  = "Final Active Set Percentage",
              xlabel = L"$\log_{10} \,\rho$",
              ylabel = L"$\log_{10} \,\eta$"
             )
heatmap!(active, log10.(rho_range), log10.(eta_range), cceqr_active)
Colorbar(fig[2,2], limits = extrema(cceqr_active))

cycles = Axis(fig[2,3],
              title  = "CCEQR Cycle Count",
              xlabel = L"$\log_{10} \,\rho$",
              ylabel = L"$\log_{10} \,\eta$"
             )
heatmap!(cycles, log10.(rho_range), log10.(eta_range), cceqr_cycles)
Colorbar(fig[2,4], limits = extrema(cceqr_cycles))

save(destination*"_plot.pdf", fig)
