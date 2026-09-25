export MentalProtocol,
    MentalState,
    mparse,
    PerceptionProtocol,
    PlanningProtocol,
    AttentionProtocol,
    Agent,
    step_module!

"The algorithmic implementation of a mental process"
abstract type MentalProtocol end

"The state of a mental process"
abstract type MentalState{T<:MentalProtocol} end

"""
A mental module with a factorized implementation of protocol and state
"""
mutable struct MentalModule{T<:MentalProtocol}
    protocol::T
    state::MentalState{T}
end

function mparse(m::MentalModule{T})::Tuple{T, MentalState{T}} where {T}
    (m.protocol, m.state)
end

"Converts sensory signals to representations of the world"
abstract type PerceptionProtocol <: MentalProtocol end
"Digests perceptual inferences to determine rewarding actions"
abstract type PlanningProtocol <: MentalProtocol end
"Rations resources within inference procedures"
abstract type AttentionProtocol <: MentalProtocol end

# abstract type PerceptionModule{T<:PerceptionProtocol} <:MentalModule{T} end
# abstract type PlanningModule{T<:PlanningProtocol} <:MentalModule{T} end
# abstract type AttentionModule{T<:AttentionProtocol} <:MentalModule{T} end

function step_module! end

mutable struct Agent{V<:PerceptionProtocol,
                     P<:PlanningProtocol,
                     A<:AttentionProtocol}
    "Observations -> Worlds"
    perception::MentalModule{V}
    "Worlds -> Goals"
    planning::MentalModule{P}
    "What to attend to in the world"
    attention::MentalModule{A}
end

include("perception/perception.jl")
include("planning/planning.jl")
include("attention/attention.jl")
