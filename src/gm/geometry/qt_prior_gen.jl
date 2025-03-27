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
        w = exp2(2 * (1 - level(n)))
        # mu = @trace(beta_uniform(w, .75, .75), :mu)
        mu ~ beta_mixture(w, 1.0, 4.0, 0.1, 0.1)
        # mu = @trace(uniform(0., 1.0), :mu)
        # mu = @trace(beta(1.0, 2.0), :mu)
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
