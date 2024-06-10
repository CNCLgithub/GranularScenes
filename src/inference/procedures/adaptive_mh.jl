export AdaptiveMH,
    AMHChain

#################################################################################
# Attention MCMC
#################################################################################

@with_kw struct AdaptiveMH <: Gen_Compose.MCMC

    #############################################################################
    # Inference Paremeters
    #############################################################################

    # chain length
    samples::Int64 = 10


    #############################################################################
    # Data driven proposal
    #############################################################################

    ddp::Function = ddp_init_kernel
    ddp_args::Tuple


    #############################################################################
    # MCMC block
    #############################################################################

    rw_budget::Int64 = 10
    sm_budget::Int64 = 10

    #############################################################################
    # Attention
    #############################################################################
    protocol::AttentionProtocol
end

# function load(::Type{AdaptiveMH}, path::String; kwargs...)
#     loaded = read_json(path)
#     AdaptiveMH(; loaded...,
#                 kwargs...)
# end


const AMHChain = Gen_Compose.MHChain{StaticQuery, AdaptiveMH}

function Gen_Compose.initialize_chain(proc::AdaptiveMH,
                                      query::StaticQuery,
                                      n::Int)
    # Intialize using DDP
    cm = query.observations
    tracker_cm = generate_qt_from_ddp(proc.ddp_args...)
    set_submap!(cm, :trackers,
                get_submap(tracker_cm, :trackers))
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           cm)
    # initialize auxillary state
    aux = AuxState(proc.protocol, trace)
    # initialize chain
    AMHChain(query, proc, trace, aux, 1, n)
end


function Gen_Compose.step!(chain::AMHChain)

    @debug "mc step $(i)"

    # proposal
    kernel_move!(chain)

    viz_chain(chain)
    println("current score $(get_score(chain.state))")
    return nothing
end

#################################################################################
# Helpers
#################################################################################

function kernel_move!(chain::AMHChain)
    state = estimate(chain)
    proc = estimator(chain)
    aux = auxillary(chain)
    @unpack protocol, rw_budget, sm_budget = proc

    # current trace
    t = state

    # select node to rejuv
    node = select_node(protocol, aux)

    rw_block_init!(aux, state)
    # RW moves - first stage
    for j = 1:rw_budget
        _t, alpha = rw_move(t, node)
        rw_block_inc!(aux, _t, alpha)
        if log(rand()) < alpha # accept?
            @show alpha
            rw_block_accept!(aux, _t)
            t = _t
        end
    end

    # # if RW acceptance ratio is high, add more
    # # otherwise, ready for SM
    # accept_ratio = max(0.25, accepts(aux) / rw_budget)
    # # TODO: integrate delta pi?
    # addition_rw_cycles =  floor(Int64, sm_budget * accept_ratio)

    # for j = 1:addition_rw_cycles
    #     _t, alpha = rw_move(t, node)
    #     rw_block_inc!(aux, _t, alpha)
    #     if log(rand()) < alpha
    #         rw_block_accept!(aux, _t)
    #         t = _t
    #     end
    # end

    rw_block_complete!(aux, protocol, t, node)

    sm_block_init!(aux, protocol)

    # SM moves
    remaining_sm = sm_budget # - addition_rw_cycles
    accept_ct = 0
    if can_split(t, node)
        is_balanced = balanced_split_merge(t, node)
        moves = is_balanced ? [split_move, merge_move] : [split_move]
        for i = 1 : remaining_sm
            move = rand(moves)
            _t, _w = split_merge_move(t, node, move)
            for _ = 1:3
                __t, __w = rw_move(move, _t, node)
                w = _w + __w
                @show w
                if log(rand()) < w
                    sm_block_accept!(aux, __t, node, move)
                    sm_block_complete!(aux, protocol, __t, node, move)
                    t = __t
                    break
                end
            end
        end

    end

    # update trace
    chain.state = t
    chain.auxillary = aux
    return nothing
end


function viz_chain(chain::AMHChain)
    @unpack auxillary, state = chain
    params = first(get_args(state))
    qt = get_retval(state)
    # println("Attention")
    # s = size(auxillary.sensitivities)
    # display_mat(reshape(auxillary.weights, s))
    if chain.step % 10 == 0
        println("Inferred state")
        display_mat(project_qt(qt))
    end
    # println("Estimated path")
    # path = Matrix{Float64}(ex_path(chain))
    # display_mat(path)
    # println("Predicted Image")
    # display_img(trace_st.img_mu)
end

# function display_selected_node(sidx, dims)
#     bs = zeros(dims)
#     bs[sidx] .= 1
#     println("Selected node")
#     display_mat(bs)
# end
