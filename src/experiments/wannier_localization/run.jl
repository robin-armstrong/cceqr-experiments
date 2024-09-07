using LinearAlgebra
using StatsBase
using JLD2

include("../../algorithms/cceqr.jl")

##########################################################################
######################## SCRIPT PARAMETERS ###############################
##########################################################################

molecule  = "alkane"     # either "water" or "alkane"

rho_range = exp10.(range(-4, -1, 5))
eta_range = exp10.(range(-4, -1, 5))
numtrials = 1

plot_only   = false     # if "true" then data will be read from disk and not regenerated
destination = "src/experiments/wannier_localization/test"
readme      = "Getting the Wannier localization script to work."

##########################################################################
######################## DATA GENERATION #################################
##########################################################################

if(!plot_only)
    # logging the parameters used for this experiment

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
        open(logfile, "w")
        write(logfile, logstr)
        close(logfile)

        fprintln(logstr)

        cceqr_time    = zeros(length(rho_range), length(eta_range), numtrials)
        cceqr_cycles  = zeros(length(rho_range), length(eta_range))
        cceqr_avgblk  = zeros(length(rho_range), length(eta_range))
        cceqr_active  = zeros(length(rho_range), length(eta_range))
        geqp3_time    = zeros(length(rho_range), length(eta_range), numtrials)

        totaltrials = length(rho_range)*length(eta_range)*numtrials
        trialcount  = 0

        fprintln("loading Wannier basis...")
        @load "data/matrices/"*molecule*".jld2" Psi Ls Ns
        
        m, n = size(Psi)
        Vt   = zeros(m, n)

        for rho_index = 1:length(rho_range)
            rho = rho_range[rho_index]

            for eta_index = 1:length(eta_range)
                eta = eta_range[eta_index]

                copy!(Vt, Psi)
                qrobj   = qr!(Vt, ColumnNorm())
                p_geqp3 = qrobj.p[1:m]

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
                    t = @elapsed qr!(Vt, ColumnNorm())
                    geqp3_time[rho_index, eta_index, trial_index] = t

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
