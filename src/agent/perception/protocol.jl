export AdaptiveMH,
    AMHChain,
    maximum_aposteriori

@kwdef struct AdaptiveMH <: PerceptionProtocol

    model::GenerativeFunction
    model_args::Tuple
    obs::ChoiceMap

    chain_length::Int = 100

    burn_in::Int = 100

    ddp::Union{Nothing, GenerativeFunction} = nothing
    ddp_args::Tuple = ()

    rw_budget::Int64 = 10
end

mutable struct AMHChain{T<:Gen.Trace} <: MentalState{AdaptiveMH}
    samples::CircularBuffer{T}
    weights::CircularBuffer{Float64}
end

function AMHChain(proc::AdaptiveMH,
                  obs::ChoiceMap)

    # Optionally initialize constraints based of DDP
    if !isnothing(proc.ddp)
        constraints = proc.ddp(proc.ddp_args...)
        constraints = merge!(obs)
    else
        constraints = obs
    end
    
    # Sample initial trace
    trace, _ = Gen.generate(proc.model,
                            proc.model_args,
                            constraints)

    # Store in chain
    T = QTVisionTrace
    samples = CircularBuffer{T}(proc.chain_length)
    push!(samples, trace)

    weights = CircularBuffer{Float64}(proc.chain_length)
    push!(weights, 1.0)

    AMHChain{T}(samples, weights)
end

function PerceptionModule(prot::AdaptiveMH, obs::ChoiceMap)
    MentalModule(prot, AMHChain(prot, obs))
end

function step_module!(perception::MentalModule{<:AdaptiveMH})

    proc, state = mparse(perception)

    # Pre-attentive scene processing
    for _ = 1:proc.rw_budget
        trace = state.samples[end]
        new_trace, w = select_node_uniform(trace)
        # MH acceptance function
        if log(rand()) < w
            # Add new trace to chain
            push!(state.samples, new_trace)
            # increment weight
            push!(state.weights, state.weights[end] + w)
        end
    end
    return nothing
end

function maximum_aposteriori(perception::MentalModule{<:AdaptiveMH})
    proc, state = mparse(perception)
    idx = argmax(state.weights)
    state.samples[idx]
end
