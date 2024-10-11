export quad_tree_prior

using Statistics: mean

@gen (static) function qt_production(n::QTProdNode)
    w = produce_weight(n)
    s = @trace(bernoulli(w), :produce)
    children::Vector{QTProdNode} = s ? produce_qt(n) : QTProdNode[]
    result = Production(n, children)
    return result
end

@gen function qt_aggregation(n::QTProdNode,
                             children::Vector{QTAggNode})
    local mu
    if isempty(children)
        # w = 1.0 - exp2(-0.5*n.level + 0.5)
        # mu = @trace(beta_uniform(w, 0.5, 0.5), :mu)
        # mu = @trace(uniform(0., 1.0), :mu)
        mu = @trace(beta(0.8, 4.0), :mu)
    else
        mu = mean(weight, children)
    end

    agg::QTAggNode = QTAggNode(n, mu, children)
    return agg
end

const quad_tree_prior = Recurse(qt_production,
                                qt_aggregation,
                                4, # quad tree only can have 4 children
                                QTProdNode,# U (production to children)
                                QTProdNode,# V (production to aggregation)
                                QTAggNode) # W (aggregation to parents)
