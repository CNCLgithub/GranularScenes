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

    ddp::Function = (_...) -> choicemap() # generate_cm_from_ddp
    ddp_args::Tuple = ()


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

const AMHChain = Gen_Compose.MHChain{StaticQuery, AdaptiveMH}

function Gen_Compose.initialize_chain(proc::AdaptiveMH,
                                      query::StaticQuery,
                                      n::Int)
    # Intialize using DDP
    cm = query.observations
    prior_cm = proc.ddp(proc.ddp_args...)
    if has_submap(prior_cm, :trackers)
        set_submap!(cm, :trackers,
                    get_submap(prior_cm, :trackers))
    end
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
        println("RW weight: $(alpha)")
        compare_latents(t, _t, node)
        rw_block_inc!(aux, _t, node)
        if log(rand()) < alpha # accept?
            # @show alpha
            rw_block_accept!(aux, _t, node)
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
    if can_split(t, node)
        is_balanced = balanced_split_merge(t, node)
        moves = is_balanced ? [split_move, merge_move] : [split_move]
        for i = 1 : remaining_sm
            move = rand(moves)
            _t, _w = split_merge_move(t, node, move)
            println("$(move) weight: $(_w)")
            compare_latents(t, _t, move, node)

            if log(rand()) < _w
                sm_block_accept!(aux, _t, node, move)
                sm_block_complete!(aux, protocol, _t, node, move)
                t = _t
                break
            end
            # __t = inner_rw_moves!(aux, protocol, _t, _w, move, node)
            # if accepts(aux) > 0
            #     t = __t
            #     break
            # end
        end
    end

    # update trace
    chain.state = t
    chain.auxillary = aux
    return nothing
end

function inner_rw_moves!(aux, p, t, w, m, i, steps = 3)
    for _ = 1:steps
        _t, _w = rw_move(m, t, i)
        __w = w + _w
        println("$(m) + RW weight: $(_w)")
        if log(rand()) < __w
            sm_block_accept!(aux, _t, i, m)
            sm_block_complete!(aux, p, _t, i, m)
            return _t
        end
    end
    return t
end


function viz_chain(chain::AMHChain)
    @unpack auxillary, state = chain
    params = first(get_args(state))
    qt = get_retval(state)
    # println("Attention")
    # s = size(auxillary.sensitivities)
    # display_mat(reshape(auxillary.weights, s))
    # if chain.step % 10 == 0
    println("Inferred state")
    display_mat(project_qt(qt))
    # end
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
