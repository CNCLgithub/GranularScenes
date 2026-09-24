export QTVisionPrior, QTVisionLikelihood, QTVisionTrace


#################################################################################
# Model specification
#################################################################################

"""
Parameters for occupancy weights
"""
@with_kw struct QTVisionPrior
    "Variance: k_0*2^(D-1)" 
    k0::Float64 = 1.0
    "Minimum lambda term for occupancy mean"
    lambda_min::Float64 = 0.01
    "Average lambda term for occupancy mean"
    lambda_0::Float64 = 1.0
end

"""
Parameterizes the depth-map likelihood
"""
@with_kw struct QTVisionLikelihood
    "Depth-render forward function"
    renderer::QuadTreeRenderer
    "minimum variance in prediction"
    pixel_var::Float32 = 0.01
end

include("qt_model_gen.jl")

const QTVisionTrace = Gen.get_trace_type(qt_vision)
