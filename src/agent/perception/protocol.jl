export AdaptiveMH,
    AMHChain

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
    trace = Gen.simulate(proc.model,
                         proc.model_args,
                         constraints)

    # Store in chain
    T = typeof{trace}
    samples = CircularBuffer{T}(proc.chain_length)
    push!(samples, trace)

    weights = CircularBuffer{Float64}(proc.chain_length)
    push!(weights, 1.0)

    AMHChain(samples, weights)
end

function step_module!(perception::MentalModule{V<:AdaptiveMH})

    proc, state = mparse(perception)

    # Pre-attentive scene processing
    for _ = 1:proc.rw_budget
        trace = state.samples[end]
        selected = select_uniform(trace)
        new_trace, w, _... = regenerate(trace, selected)

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
