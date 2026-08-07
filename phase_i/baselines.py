import numpy as np
import scipy.stats as stats

def run_naive_frequency(scenario):
    """
    Baseline A: Naive Frequency
    Score = num_failed_inspections / total_inspections involving each factory
    """
    N = scenario['num_factories']
    store_outcomes = scenario['store_outcomes']  # 1 for pass, 0 for fail
    
    # Track inspections for each factory
    failed_counts = np.zeros(N)
    total_counts = np.zeros(N)
    
    for edge in scenario['edges']:
        fac_idx = edge['factory_idx']
        store_idx = edge['store_idx']
        
        total_counts[fac_idx] += 1
        if store_outcomes[store_idx] == 0:
            failed_counts[fac_idx] += 1
            
    # Compute scores
    scores = np.zeros(N)
    for i in range(N):
        if total_counts[i] > 0:
            scores[i] = failed_counts[i] / total_counts[i]
        else:
            scores[i] = 0.0  # Default if no inspections
            
    # Estimated quality: 1.0 - score (scaled to represent reasonable qualities)
    # E.g. if failure rate is 0, quality is 0.98. If failure rate is 1, quality is 0.80.
    est_qualities = 0.98 - 0.18 * scores
    
    return est_qualities, scores

def run_failure_only_frequency(scenario):
    """
    Baseline B: Failure-Only Frequency
    Score = num_failed_inspections involving each factory / total failed inspections in graph
    """
    N = scenario['num_factories']
    store_outcomes = scenario['store_outcomes']
    
    total_failures = np.sum(store_outcomes == 0)
    
    failed_counts = np.zeros(N)
    for edge in scenario['edges']:
        fac_idx = edge['factory_idx']
        store_idx = edge['store_idx']
        
        if store_outcomes[store_idx] == 0:
            failed_counts[fac_idx] += 1
            
    scores = np.zeros(N)
    if total_failures > 0:
        scores = failed_counts / total_failures
        
    # Estimate quality: map score to quality range
    est_qualities = 0.98 - 0.18 * scores
    
    return est_qualities, scores

def run_constraint_elimination(scenario):
    """
    Baseline C: Constraint-Based Elimination
    Assume every factory is suspect (score = 1) until a store it sends product to passes inspection (score = 0)
    """
    N = scenario['num_factories']
    store_outcomes = scenario['store_outcomes']
    
    # Start as suspect
    suspect = np.ones(N)
    
    for edge in scenario['edges']:
        fac_idx = edge['factory_idx']
        store_idx = edge['store_idx']
        
        if store_outcomes[store_idx] == 1:
            suspect[fac_idx] = 0.0  # Eliminated because a store it supplied passed
            
    # Estimated qualities: 0.98 if eliminated, 0.85 if suspect
    est_qualities = np.where(suspect == 0.0, 0.98, 0.85)
    
    return est_qualities, suspect

def run_bayesian_inference(scenario, num_iter=300, burn_in=100, p_fail_prior=0.10):
    """
    Baseline D: Bayesian Inference using Gibbs Sampling over a discrete state space
    State g_i in {0, 1}: 1 = good (q = 0.98), 0 = bad (q = 0.85)
    """
    N = scenario['num_factories']
    M = scenario['num_stores']
    store_outcomes = scenario['store_outcomes']
    edges = scenario['edges']
    
    # 1. Build adjacency structures
    # store_to_edges[j] = list of edges connected to store j
    store_to_edges = [[] for _ in range(M)]
    # factory_to_stores[i] = list of store indices connected to factory i
    factory_to_stores = [[] for _ in range(N)]
    
    for edge_idx, edge in enumerate(edges):
        fac_idx = edge['factory_idx']
        store_idx = edge['store_idx']
        store_to_edges[store_idx].append(edge)
        factory_to_stores[fac_idx].append(store_idx)
        
    # 2. Initialize factory states g_i
    # Use constraint-based elimination to get a very good initial state
    _, suspect = run_constraint_elimination(scenario)
    g = 1.0 - suspect  # 1 for clean, 0 for suspect
    
    # Pre-calculate store sample sizes
    store_samples = np.zeros(M)
    for j in range(M):
        store_samples[j] = sum(edge['num_samples'] for edge in store_to_edges[j])
        
    # Quality levels
    q_good = 0.98
    q_bad = 0.85
    
    # Define likelihood function for a single store j given current factory states g
    def compute_store_likelihood(store_idx, current_g):
        total_n = store_samples[store_idx]
        if total_n == 0:
            return 1.0
            
        # Expected quality at store j: weighted average of qualities of supplying factories
        sum_n_q = 0.0
        for edge in store_to_edges[store_idx]:
            fac_idx = edge['factory_idx']
            n_fac = edge['num_samples']
            fac_q = q_good if current_g[fac_idx] == 1 else q_bad
            sum_n_q += n_fac * fac_q
            
        q_j = sum_n_q / total_n
        
        # Inspection passes if successes / total_n >= 0.95
        # So k_pass = ceil(0.95 * total_n)
        k_pass = int(np.ceil(0.95 * total_n))
        
        # Probability that binomial variable X >= k_pass
        # sf(x, n, p) = P(X > x) = P(X >= x + 1)
        # So P(X >= k_pass) = sf(k_pass - 1, total_n, q_j)
        p_pass = stats.binom.sf(k_pass - 1, total_n, q_j)
        # Clip to avoid exact 0 or 1
        p_pass = np.clip(p_pass, 1e-6, 1.0 - 1e-6)
        
        if store_outcomes[store_idx] == 1:
            return p_pass
        else:
            return 1.0 - p_pass

    # Samples container
    g_samples = []
    
    # 3. Gibbs Sampling loop
    for iteration in range(num_iter):
        for i in range(N):
            # Compute likelihoods if g[i] = 1 vs g[i] = 0
            # We only need to check stores connected to factory i
            connected_stores = factory_to_stores[i]
            
            # g_i = 1
            g[i] = 1.0
            L1 = 1.0
            for s_idx in connected_stores:
                L1 *= compute_store_likelihood(s_idx, g)
                
            # g_i = 0
            g[i] = 0.0
            L0 = 1.0
            for s_idx in connected_stores:
                L0 *= compute_store_likelihood(s_idx, g)
                
            # Posteriors
            post1 = L1 * (1.0 - p_fail_prior)
            post0 = L0 * p_fail_prior
            
            # Normalize
            prob_good = post1 / (post1 + post0 + 1e-12)
            
            # Sample new state
            g[i] = 1.0 if np.random.rand() < prob_good else 0.0
            
        if iteration >= burn_in:
            g_samples.append(g.copy())
            
    # 4. Compute posterior mean probabilities
    g_samples = np.array(g_samples)
    prob_good = np.mean(g_samples, axis=0)  # Shape (N,)
    prob_fail = 1.0 - prob_good
    
    # Estimated quality is the expected quality under the posterior
    est_qualities = prob_good * q_good + prob_fail * q_bad
    
    return est_qualities, prob_fail

if __name__ == '__main__':
    # Simple test of baselines
    from data_generator import generate_company_scenario
    
    scenario = generate_company_scenario(
        num_factories=10,
        num_fails=2,
        num_stores=5,
        avg_factories_per_store=3.0,
        sample_size=100
    )
    
    print("True Qualities:", np.round(scenario['true_qualities'], 3))
    print("True Fails:", scenario['is_failed_factory'].astype(int))
    
    q_A, p_A = run_naive_frequency(scenario)
    print("Baseline A Est Qualities:", np.round(q_A, 3))
    
    q_B, p_B = run_failure_only_frequency(scenario)
    print("Baseline B Est Qualities:", np.round(q_B, 3))
    
    q_C, p_C = run_constraint_elimination(scenario)
    print("Baseline C Est Qualities:", np.round(q_C, 3))
    
    q_D, p_D = run_bayesian_inference(scenario)
    print("Baseline D Est Qualities:", np.round(q_D, 3))
    print("Baseline D Failure Probs:", np.round(p_D, 3))
