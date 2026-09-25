export AStarPlanner, AStarState, PlanningModule

@kwdef struct AStarPlanner <: PlanningProtocol
    ent::Int
    ext::Int
    nsamples::Int = 100
    obstacle_cost::Float64 = 1.0
    grad_tau::Float64 = 1.0
end


mutable struct AStarState <: MentalState{AStarPlanner}
    scores::Vector{Float64}
    paths::Vector{Vector{NodeId}}
end


function PlanningModule(protocol::AStarPlanner)
    scores = Vector{Float64}(undef, protocol.nsamples)
    fill!(scores, -Inf)
    paths = Vector{Vector{NodeId}}(undef, protocol.nsamples)
    state = AStarState(scores, paths)
    MentalModule{AStarPlanner}(protocol, state)
end


function step_module!(planning::MentalModule{<:AStarPlanner},
                      perception::MentalModule{<:AdaptiveMH})
    protocol, state = mparse(planning)

    _, vstate = mparse(perception)

    mass = logsumexp(vstate.weights)

    n = length(vstate.samples)
    for i = 1:n
        vtrace = vstate.samples[i]
        vweight = vstate.weights[i] - mass
        qt = get_retval(vtrace)
        score, path, dpi =
            qt_a_star(qt,
                      protocol.obstacle_cost,
                      protocol.ent, protocol.ext,
                      protocol.grad_tau)

        state.scores[i] = -score + vweight
        state.paths[i] = path
    end

    return nothing
end


function best_path(planning::MentalModule{<:AStarPlanner})
    protocol, state = mparse(planning)
    idx = argmax(state.scores)
    return state.paths[idx]
end
