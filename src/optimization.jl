# --- OPTIMIZATION FUNCTIONS ---

"""
    optimize_equivalent_capacities(

        ttc_original::DataFrame,
        ptdf_reduced_results::DataFrame;
        Type::String = "QP",
        bigM_factor::Float64 = 1000.0,
        lambda::Float64 = 0.1
    )

Optimize equivalent capacities for the reduced network using QP, MIQP, or LP.

This function solves an optimization problem to find synthetic line capacities
for the reduced network that best match the original network's TTC values.

# Arguments
- `ttc_original::DataFrame`: Original network TTC values
- `ptdf_reduced_results::DataFrame`: PTDF results from reduced network
- `Type::String`: Optimization type: "QP", "MIQP", or "LP"
- `bigM_factor::Float64`: Factor for big-M constraints (MIQP only)
- `lambda::Float64`: Regularization parameter for capacity smoothing

# Returns
- `equivalent_capacities::DataFrame`: Optimized synthetic line capacities

# Mathematical Formulation

## QP/LP Formulation:
Minimize: Σ(TTC_eq[t] - TTC_orig[t])² + λ*Σ(C_eq[l]²)  (QP)
or
Maximize: Σ TTC_eq[t] (LP with TTC_eq[t] ≤ TTC_orig[t])
Subject to: TTC_eq[t] ≤ C_eq[l] / |PTDF[t,l]| for all transactions t and lines l

## MIQP Formulation:
Minimize: Σ(TTC_eq[t] - TTC_orig[t])² + λ*Σ(C_eq[l]²)
Subject to:
    TTC_eq[t] = Σ Z[t,l] / |PTDF[t,l]|
    Z[t,l] ≤ C_eq[l] for all t,l
    Exactly one Z[t,l] = C_eq[l] per transaction t (others = 0)
    Each line must bind at least one transaction
"""
function optimize_equivalent_capacities(
    ttc_original::DataFrame,
    ptdf_reduced_results::DataFrame;
    Type::String = CONFIG.optimization_type,    # default from config
    lambda::Float64 = CONFIG.lambda
)
    println("\n--- OPTIMIZING EQUIVALENT CAPACITIES (Type: $Type) ---")

    # ------------------------------------------------------------
    # 1. DATA PREPARATION (COMMON FOR ALL MODELS)
    # ------------------------------------------------------------

    # Get unique synthetic lines
    synth_lines_df = unique(ptdf_reduced_results, [:synth_line_from, :synth_line_to])
    
    synth_lines = Tuple{Int,Int}[]
    synth_line_map = Dict{Tuple{Int,Int},Int}()
    
    for row in eachrow(synth_lines_df)
        u, v = min(row.synth_line_from, row.synth_line_to),
               max(row.synth_line_from, row.synth_line_to)
        key = (u, v)
        if !haskey(synth_line_map, key)
            push!(synth_lines, key)
            synth_line_map[key] = length(synth_lines)
        end
    end
    
    L = 1:length(synth_lines)  # Synthetic line indices
    
    # Get reduced network bus IDs
    rn_ids = unique(vcat(
        ptdf_reduced_results.transaction_from_orig,
        ptdf_reduced_results.transaction_to_orig,
    ))
    
    # Filter canonical transactions (from reduced network only)
    ttc_canonical = filter(
        r -> (r.transaction_from in rn_ids) &&
             (r.transaction_to in rn_ids) &&
             (r.transaction_from < r.transaction_to),
        ttc_original,
    )
    
    TR = 1:nrow(ttc_canonical)  # Transaction indices
    
    println("Synthetic lines: $(length(L))")
    println("Canonical transactions: $(length(TR))")
    
    # Create PTDF lookup dictionary: (t,l) → |PTDF|
    PTDF = Dict{Tuple{Int,Int},Float64}()
    epsilon = CONFIG.ptdf_epsilon
    
    for (t, row) in enumerate(eachrow(ttc_canonical))
        a, b = row.transaction_from, row.transaction_to
        
        # Get all PTDF entries for this transaction
        txn_rows = filter(
            r -> r.transaction_from_orig == a &&
                 r.transaction_to_orig == b,
            ptdf_reduced_results,
        )
        
        for r in eachrow(txn_rows)
            # Use canonical line ordering
            lkey = (min(r.synth_line_from, r.synth_line_to),
                    max(r.synth_line_from, r.synth_line_to))
            
            if haskey(synth_line_map, lkey)
                l = synth_line_map[lkey]
                val = abs(r.PTDF_value)
                if val > epsilon
                    PTDF[(t,l)] = val
                end
            end
        end
    end
    
    ttc_orig = ttc_canonical.TTC_pu  # Original TTC values
    
    # Precompute relevant lines for each transaction
    relevant_lines_per_txn = [Int[] for _ in TR]
    for t in TR
        relevant_lines_per_txn[t] = [l for l in L if haskey(PTDF, (t,l))]
    end
    
    # Precompute transactions for each line
    relevant_txns_per_line = [Int[] for _ in L]
    for t in TR
        for l in relevant_lines_per_txn[t]
            push!(relevant_txns_per_line[l], t)
        end
    end
    
    # ------------------------------------------------------------
    # 2. MODEL SELECTION AND SETUP
    # ------------------------------------------------------------
    
    if Type == "QP"
        model, TTC_vals = _solve_qp_model(ttc_orig, synth_lines, L, TR, PTDF, 
                                         relevant_lines_per_txn, lambda)
    
    elseif Type == "MIQP"
        model, TTC_vals = _solve_miqp_model(ttc_orig, synth_lines, L, TR, PTDF,
                                           relevant_lines_per_txn, relevant_txns_per_line,
                                           lambda)
    
    elseif Type == "LP"
        model, TTC_vals = _solve_lp_model(ttc_orig, synth_lines, L, TR, PTDF,
                                         relevant_lines_per_txn)
    
    else
        error("Invalid Type: $Type. Choose from: QP, MIQP, LP")
    end

    # ------------------------------------------------------------
    # 3. EXTRACT BOTH CAPACITIES AND TTC RESULTS
    # ------------------------------------------------------------
    
    # Extract equivalent capacities
    equivalent_capacities = _extract_capacities(model, synth_lines, L, TR, ttc_orig, Type)
    
    # Create TTC results directly from optimization variables
    ttc_equivalent_results = create_ttc_results_from_optimization(
        TTC_vals,
        ttc_canonical,
        model = model,
        Type = Type,
        synth_lines = synth_lines
    )
    
    return equivalent_capacities, ttc_equivalent_results
end

# ------------------------------------------------------------
# QP MODEL IMPLEMENTATION
# ------------------------------------------------------------

function _solve_qp_model(
    ttc_orig::Vector{Float64},
    synth_lines::Vector{Tuple{Int,Int}},
    L::UnitRange{Int},
    TR::UnitRange{Int},
    PTDF::Dict{Tuple{Int,Int},Float64},
    relevant_lines_per_txn::Vector{Vector{Int}},
    lambda::Float64
)
    println("Setting up QP model...")
    
    model = Model(CPLEX.Optimizer)
    set_silent(model)
   
    # set_attribute(model, "CPX_PARAM_EPGAP", 0.01) # 1% gap
    # Memory Emphasis: Tells CPLEX to conserve memory where possible
    # set_attribute(model, "CPXPARAM_Emphasis_Memory", 1)
    # WorkDir: If CPLEX hits the RAM limit, it will write temporary 
    # "Node Files" to disk instead of crashing.
    # set_attribute(model, "CPXPARAM_WorkDir", "C:/temp/cplex_work")
    # set_attribute(model, "CPXPARAM_MIP_Strategy_File", 3) # 3 = Compress and write to disk
    
    # Variables
    @variable(model, C_eq[l in L] >= 0)           # Equivalent capacities
    @variable(model, TTC_eq[t in TR] >= 0)        # Equivalent TTCs
    @variable(model, V_mismatch[t in TR])         # TTC mismatches
    
    # Objective: Minimize squared error with regularization
    @objective(model, Min, 
        sum(V_mismatch[t]^2 for t in TR) + lambda * sum(C_eq[l]^2 for l in L)
    )
    
    # Constraints
    # 1. Mismatch definition
    @constraint(model, [t in TR],
        V_mismatch[t] == TTC_eq[t] - ttc_orig[t]
    )
    
    # 2. Physical constraints: TTC cannot exceed capacity/PTDF ratio
    for t in TR
        for l in relevant_lines_per_txn[t]
            @constraint(model,
                TTC_eq[t] <= C_eq[l] / PTDF[(t,l)]
            )
        end
    end
    
    # Solve
    optimize!(model)
    
    status = termination_status(model)
    println("QP status: $status")
    
    if status != MathOptInterface.OPTIMAL
        println("Warning: QP solution status: $status")
    end
    
    # Extract results
    TTC_vals_raw = value.(model[:TTC_eq])
    TTC_vals = [TTC_vals_raw[t] for t in TR]

    return model, TTC_vals
end

# ------------------------------------------------------------
# MIQP MODEL IMPLEMENTATION
# ------------------------------------------------------------

function _solve_miqp_model(
    ttc_orig::Vector{Float64},
    synth_lines::Vector{Tuple{Int,Int}},
    L::UnitRange{Int},
    TR::UnitRange{Int},
    PTDF::Dict{Tuple{Int,Int},Float64},
    relevant_lines_per_txn::Vector{Vector{Int}},
    relevant_txns_per_line::Vector{Vector{Int}},
   # bigM_factor::Float64,
    lambda::Float64
)
    println("Setting up MIQP model...")
    
    # Compute big-M values
    #  max_ttc = maximum(ttc_orig)
    # bigM = bigM_factor * max_ttc

    # Alternate selection for big-M
    max_ptdf_val = maximum(abs.(values(PTDF)))
    max_ttc = maximum(ttc_orig)
    bigM = max_ttc * max_ptdf_val * 5.0  # Safety factor of 10
    
    model = Model(CPLEX.Optimizer)
    set_silent(model)
    
    # Variables
    @variable(model, C_eq[l in L] >= 0)           # Equivalent capacities
    @variable(model, TTC_eq[t in TR] >= 0)        # Equivalent TTCs
    @variable(model, V_mismatch[t in TR])         # TTC mismatches
    @variable(model, Z[t in TR, l in L] >= 0)     # Capacity allocation
    @variable(model, b[t in TR, l in L], Bin)     # Binding indicator
    
    # Objective: Minimize squared error with regularization
    @objective(model, Min, 
        sum(V_mismatch[t]^2 for t in TR) + lambda * sum(C_eq[l]^2 for l in L)
    )
    
    # Constraints
    
    # 1. Mismatch definition
    @constraint(model, [t in TR],
        V_mismatch[t] == TTC_eq[t] - ttc_orig[t]
    )
    
    # 2. TTC definition as sum of allocated capacities
    for t in TR
        relevant_lines = relevant_lines_per_txn[t]
        if isempty(relevant_lines)
            @constraint(model, TTC_eq[t] == 0)
        else
            @constraint(model,
                TTC_eq[t] == sum(Z[t,l] / PTDF[(t,l)] for l in relevant_lines)
            )
        end
    end
    
    # 3. Capacity allocation cannot exceed line capacity
    for t in TR
        for l in relevant_lines_per_txn[t]
            @constraint(model,
                Z[t,l] <= C_eq[l])
        end
    end
    
    # 4. Big-M constraints for binding indicator
    for t in TR
        for l in relevant_lines_per_txn[t]
            # If b[t,l] = 1, then Z[t,l] = C_eq[l]
            # If b[t,l] = 0, then Z[t,l] = 0
            @constraint(model,
                C_eq[l] - Z[t,l] <= bigM * (1 - b[t,l]))
            @constraint(model,
                Z[t,l] <= bigM * b[t,l])
        
        end
    end
    
    # 5. Exactly one line binds per transaction (disjunctive constraint)
    for t in TR
        relevant_lines = relevant_lines_per_txn[t]
        if !isempty(relevant_lines)
            @constraint(model,
                sum(b[t,l] for l in relevant_lines) <= 1)
        end
    end

    # 6. Deactivate variables for irrelevant (t,l) pairs
    for t in TR
        for l in L
            if !haskey(PTDF, (t,l))
                fix(b[t,l], 0; force=true)
                fix(Z[t,l], 0; force=true)
            end
        end
    end
    
    # Solve
    println("Solving MIQP...")
    optimize!(model)
    
    status = termination_status(model)
    println("MIQP status: $status")
    
    if status != MathOptInterface.OPTIMAL && 
       status != MathOptInterface.LOCALLY_SOLVED
        println("Warning: MIQP solution status: $status")
    end
    
    # Extract results
    TTC_vals_raw = value.(model[:TTC_eq])
    TTC_vals = [TTC_vals_raw[t] for t in TR]
    return model, TTC_vals
end

# ------------------------------------------------------------
# LP MODEL IMPLEMENTATION
# ------------------------------------------------------------

function _solve_lp_model(
    ttc_orig::Vector{Float64},
    synth_lines::Vector{Tuple{Int,Int}},
    L::UnitRange{Int},
    TR::UnitRange{Int},
    PTDF::Dict{Tuple{Int,Int},Float64},
    relevant_lines_per_txn::Vector{Vector{Int}}
)
    println("Setting up LP model...")
    
    model = Model(CPLEX.Optimizer)
    set_silent(model)
    
    # Variables
    @variable(model, C_eq[l in L] >= 0)           # Equivalent capacities
    @variable(model, TTC_eq[t in TR] >= 0)        # Equivalent TTCs
    
    # Objective: Maximize total TTC (subject to TTC_eq ≤ TTC_orig)
    @objective(model, Max, sum(TTC_eq[t] for t in TR))
    
    # Constraints
    
    # 1. TTC cannot exceed original TTC
    @constraint(model, [t in TR],
        TTC_eq[t] <= ttc_orig[t]
    )
    
    # 2. Physical constraints: TTC cannot exceed capacity/PTDF ratio
    for t in TR
        for l in relevant_lines_per_txn[t]
            @constraint(model,
                TTC_eq[t] <= C_eq[l] / PTDF[(t,l)]
            )
        end
    end
    
    # Solve
    optimize!(model)
    
    status = termination_status(model)
    println("LP status: $status")
    
    if status != MathOptInterface.OPTIMAL
        println("Warning: LP solution status: $status")
    end
    
    # Extract results
    TTC_vals_raw = value.(model[:TTC_eq])
    TTC_vals = [TTC_vals_raw[t] for t in TR]
    return model, TTC_vals
end

# ------------------------------------------------------------
# RESULT EXTRACTION FUNCTION
# ------------------------------------------------------------

function _extract_capacities(
    model::Model,
    synth_lines::Vector{Tuple{Int,Int}},
    L::UnitRange{Int},
    TR::UnitRange{Int},
    ttc_orig::Vector{Float64},
    Type::String = CONFIG.optimization_type
)
    # Extract only capacities
    C_vals = value.(model[:C_eq])
    
    equivalent_capacities = DataFrame(
        synth_line_from = Int[],
        synth_line_to   = Int[],
        C_eq_pu         = Float64[],
    )
    
    for (i, (u,v)) in enumerate(synth_lines)
        push!(equivalent_capacities, (u, v, C_vals[i]))
    end
    
    # Print statistics 
    println("\n=== $Type Optimization Results ===")
    println("C_eq statistics:")
    println("  Min: $(round(minimum(C_vals); digits=6)) pu")
    println("  Max: $(round(maximum(C_vals); digits=6)) pu")
    println("  Mean: $(round(mean(C_vals); digits=6)) pu")
    println("  Std: $(round(std(C_vals); digits=6)) pu")
    
    return equivalent_capacities
end

"""
    create_ttc_results_from_optimization(
        ttc_vals::Vector{Float64},
        ttc_canonical::DataFrame;
        model::Union{Model,Nothing} = nothing,
        Type::String = "QP",
        synth_lines::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]
    )

Create TTC results directly from optimization variables.

"""
function create_ttc_results_from_optimization(
    ttc_vals::Vector{Float64},
    ttc_canonical::DataFrame;
    model = model,
    Type = CONFIG.optimization_type,
    synth_lines::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]
)

    n_txn = length(ttc_vals)
    @assert n_txn == nrow(ttc_canonical) "TTC size mismatch"

    # Base results (common for all types)
    ttc_results = DataFrame(
        transaction_from = ttc_canonical.transaction_from,
        transaction_to   = ttc_canonical.transaction_to,
        TTC_Equivalent_pu = ttc_vals
    )

    # ------------------------------------------------------------
    # MIQP: extract binding lines from b[t,l]
    # ------------------------------------------------------------
    if Type == "MIQP"
        if model === nothing
            @warn "Model not provided for MIQP - cannot extract binding lines"
            ttc_results[!, :limiting_synth_line_from] = zeros(Int, n_txn)
            ttc_results[!, :limiting_synth_line_to] = zeros(Int, n_txn)
        elseif !haskey(model.obj_dict, :b)
            @warn "MIQP model doesn't contain binary variable b[t,l]"
            ttc_results[!, :limiting_synth_line_from] = zeros(Int, n_txn)
            ttc_results[!, :limiting_synth_line_to] = zeros(Int, n_txn)
        elseif isempty(synth_lines)
            @warn "synth_lines not provided for MIQP"
            ttc_results[!, :limiting_synth_line_from] = zeros(Int, n_txn)
            ttc_results[!, :limiting_synth_line_to] = zeros(Int, n_txn)
        else
            b_vals = value.(model[:b])
            
            binding_from = Int[]
            binding_to   = Int[]
            
            for t in 1:n_txn
                l = findfirst(l -> b_vals[t,l] > 0.5, axes(b_vals,2))
                if l === nothing
                    push!(binding_from, 0)
                    push!(binding_to, 0)
                else
                    push!(binding_from, synth_lines[l][1])
                    push!(binding_to,   synth_lines[l][2])
                end
            end
            
            ttc_results[!, :limiting_synth_line_from] = binding_from
            ttc_results[!, :limiting_synth_line_to]   = binding_to
        end
    
    # ------------------------------------------------------------
    # LP/QP: leave limiting line columns empty (can calculate later if needed)
    # ------------------------------------------------------------
    else  # LP or QP
        ttc_results[!, :limiting_synth_line_from] = zeros(Int, n_txn)
        ttc_results[!, :limiting_synth_line_to] = zeros(Int, n_txn)
    end

    # Calculate error statistics
    err = ttc_vals .- ttc_canonical.TTC_pu
    println("\nTTC matching accuracy:")
    println("  Max |error| = $(round(maximum(abs.(err)); digits=6)) pu")
    println("  Mean |error| = $(round(mean(abs.(err)); digits=6)) pu")
    println("  RMS error  = $(round(sqrt(mean(err.^2)); digits=6)) pu")
    
    println("\nSample TTC comparisons (first 5 transactions):")
    for t in 1:min(5, length(ttc_vals))
        println("  T$t: Original=$(round(ttc_canonical.TTC_pu[t]; digits=6)), " *
                "Estimated=$(round(ttc_vals[t]; digits=6)), " *
                "Error=$(round(err[t]; digits=6))")
    end

    return ttc_results
end