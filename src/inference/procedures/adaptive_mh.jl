export AdaptiveMH,
    AMHChain,
    extend_chain!,
    fork

#################################################################################
# Attention MCMC
#################################################################################

@with_kw struct AdaptiveMH <: Gen_Compose.MCMC

    #############################################################################
    # Inference Paremeters
    #############################################################################

    # chain length
    # TODO: remove, not used
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
    constraints = proc.ddp(proc.ddp_args...)
    constraints[:img_a] = query.observations[:img_a]
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           constraints)

    println("Initial state")
    display_mat(project_qt(get_retval(trace)))

    # initialize auxillary state
    aux = AuxState(proc.protocol, trace)
    # initialize chain
    AMHChain(query, proc, trace, aux, 1, n)
end


function Gen_Compose.step!(chain::AMHChain)

    @debug "mc step $(i)"

    # proposal
    kernel_move!(chain)

    # viz_chain(chain)
    # println("current score $(get_score(chain.state))")
    return nothing
end

function extend_chain!(chain::AMHChain, query::StaticQuery)
    trace = estimate(chain)
    old_query = estimand(chain)
    old_args = old_query.args
    new_args = query.args
    @assert length(old_args) == length(new_args) "New query must match args"
    argdiffs = Tuple((x == y ? NoChange() : UnknownChange()
                      for (x,y) = zip(old_args, new_args)))
    new_trace, _... = update(trace, new_args, argdiffs, query.observations)
    chain.state = new_trace
    chain.query = query
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
    rw_block_init!(aux, protocol, t)
    node = select_node(protocol, aux)

    # RW moves - first stage
    for j = 1:rw_budget
        _t, alpha = rw_move(t, node)
        rw_block_inc!(aux, protocol, _t, node, alpha)
        if log(rand()) < alpha # accept?
            rw_block_accept!(aux, protocol, _t, node)
            t = _t
        end
    end

    # if RW acceptance ratio is high, add more
    # otherwise, ready for SM
    accept_ratio = accepts(aux) / rw_budget
    addition_rw_cycles =  floor(Int64, sm_budget * accept_ratio)

    for j = 1:addition_rw_cycles
        _t, alpha = rw_move(t, node)
        rw_block_inc!(aux, protocol, _t, node, alpha)
        if log(rand()) < alpha # accept?
            rw_block_accept!(aux, protocol, _t, node)
            t = _t
        end
    end
    rw_block_complete!(aux, protocol, t, node)
    sm_block_init!(aux, protocol)

    remaining_sm = sm_budget - addition_rw_cycles
    for i = 1 : remaining_sm
        _t, _w = split_merge_move(t, node)
        if log(rand()) < _w
            sm_block_accept!(aux, node)
            sm_block_complete!(aux, protocol, _t, node)
            t = _t
            break
        end
    end

    # update trace
    chain.state = t
    chain.auxillary = aux
    return nothing
end

# TODO: cleanup
function change_step!(chain::AMHChain)
    trace = estimate(chain)
    proc = estimator(chain)
    for j = 1:proc.rw_budget
        _t, alpha, _ =
            regenerate(trace,
                       select(:changes => 1 => :location))
        # println("Before $(getindex(trace, :changes => 1 => :location))")
        # println("After $(getindex(_t, :changes => 1 => :location))")
        if log(rand()) < alpha # accept?
            trace = _t
        end
    end
    chain.state = trace
end

function viz_chain(log::ChainLogger)
    bfr = buffer(log)
    println("\n\nInferred state + Path + Attention")
    geo = draw_mat(marginalize(bfr, :obstacles),
                   true, colorant"black", colorant"blue")
    # println("Estimated path")
    pth = draw_mat(marginalize(bfr, :path),
                   true, colorant"black", colorant"green")
    attm = marginalize(bfr, :attention)
    attm = softmax(attm, 0.75)
    lmul!(1.0 / maximum(attm), attm)
    att = draw_mat(attm,
                   true, colorant"black", colorant"red")
    display(reduce(hcat, [geo, pth, att]))
    # loc = marginalize(bfr, :change)
    # display_mat(loc)
    return nothing
end
function viz_chain(chain::AMHChain)
    # chain.step % 10 == 0 || return nothing
    @unpack auxillary, state = chain
    _, params = get_args(state)
    qt = get_retval(state)
    # println("Attention")
    # s = size(auxillary.sensitivities)
    # display_mat(reshape(auxillary.weights, s))
    println("\n\nInferred state + Path + Attention")
    geo = draw_mat(project_qt(qt), true, colorant"black", colorant"blue")
    # println("Estimated path")
    path = Matrix{Float64}(ex_path(chain))
    pth = draw_mat(path, true, colorant"black", colorant"green")

    attm  = ex_attention(chain)
    attm = softmax(attm, auxillary.temp)
    @show maximum(attm)
    lmul!(1.0 / maximum(attm), attm)
    att = draw_mat(attm, true, colorant"black", colorant"red")
    display(reduce(hcat, [geo, pth, att]))

    # loc = ex_loc_change(chain)
    # display_mat(loc)
    return nothing
end

function fork(c::AMHChain, l::MemLogger)
    new_c = AMHChain(
        c.query,
        c.proc,
        c.state,
        c.auxillary,
        c.step,
        c.steps,
    )
    (new_c, deepcopy(l))
end
