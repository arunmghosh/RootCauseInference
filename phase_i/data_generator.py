import numpy as np
import random

CITIES = ["Austin", "Boston", "Chicago", "Denver", "Seattle", "Dallas", "Houston", "Miami", "Atlanta", "Phoenix", "Omaha", "Detroit", "Portland", "Orlando", "Nashville", "Charlotte", "Columbus", "ElPaso", "Memphis", "Baltimore"]
ADJECTIVES = ["Apex", "Blue", "Crown", "Delta", "Echo", "Falcon", "Global", "Helix", "Iron", "Jade", "Krypton", "Lunar", "Matrix", "Nova", "Onyx", "Prime", "Quantum", "Rapid", "Summit", "Titan"]

def generate_unique_names(count, type_name):
    """
    Generates a list of unique names for factories or stores.
    """
    names = []
    for city in CITIES:
        for adj in ADJECTIVES:
            names.append(f"{city} {adj} {type_name}")
    
    rng = random.Random(42)
    rng.shuffle(names)
    
    if len(names) < count:
        for i in range(count - len(names) + 100):
            city = CITIES[i % len(CITIES)]
            adj = ADJECTIVES[(i // len(CITIES)) % len(ADJECTIVES)]
            names.append(f"{city} {adj} {type_name} #{i+1}")
            
    return names[:count]

def generate_company_scenario(num_factories, num_fails, num_stores, avg_factories_per_store, sample_size=100):
    """
    Generates a single company scenario representing factory quality and store inspections.
    """
    num_fails = max(1, min(num_fails, num_factories))
    num_stores = max(1, num_stores)
    
    failing_indices = set(random.sample(range(num_factories), num_fails))
    
    true_qualities = np.zeros(num_factories)
    for i in range(num_factories):
        if i in failing_indices:
            true_qualities[i] = np.random.uniform(0.70, 0.94)
        else:
            true_qualities[i] = np.random.uniform(0.95, 1.00)
            
    is_failed_factory = np.array([1 if i in failing_indices else 0 for i in range(num_factories)], dtype=np.float32)
    
    store_to_factories = [[] for _ in range(num_stores)]
    all_factories = list(range(num_factories))
    random.shuffle(all_factories)
    
    for i, fac_idx in enumerate(all_factories):
        store_idx = i % num_stores
        store_to_factories[store_idx].append(fac_idx)
        
    edges = []
    for store_idx in range(num_stores):
        current_facs = store_to_factories[store_idx]
        k = int(np.clip(np.random.normal(avg_factories_per_store, 1.5), 1, num_factories))
        k = max(1, min(k, num_factories))
        
        if len(current_facs) < k:
            candidates = [f for f in range(num_factories) if f not in current_facs]
            additional_facs = random.sample(candidates, k - len(current_facs))
            current_facs.extend(additional_facs)
            
        k_actual = len(current_facs)
        weights = np.random.gamma(5.0, 1.0, size=k_actual)
        weights /= weights.sum()
        
        counts = np.ones(k_actual, dtype=int)
        if sample_size > k_actual:
            extra_counts = np.random.multinomial(sample_size - k_actual, weights)
            counts += extra_counts
        else:
            counts = np.random.multinomial(sample_size, weights)
            
        for fac_idx, count in zip(current_facs, counts):
            if count > 0:
                q = true_qualities[fac_idx]
                successes = np.random.binomial(count, q)
                edges.append({
                    'factory_idx': fac_idx,
                    'store_idx': store_idx,
                    'num_samples': int(count),
                    'num_successes': int(successes)
                })
                
    store_totals = {}
    for edge in edges:
        store_idx = edge['store_idx']
        if store_idx not in store_totals:
            store_totals[store_idx] = {'samples': 0, 'successes': 0}
        store_totals[store_idx]['samples'] += edge['num_samples']
        store_totals[store_idx]['successes'] += edge['num_successes']
        
    store_outcomes = np.zeros(num_stores, dtype=np.float32)
    store_success_rates = np.zeros(num_stores, dtype=np.float32)
    store_samples = np.zeros(num_stores, dtype=np.float32)
    
    for store_idx in range(num_stores):
        if store_idx in store_totals:
            samples = store_totals[store_idx]['samples']
            successes = store_totals[store_idx]['successes']
            rate = successes / samples if samples > 0 else 1.0
            store_success_rates[store_idx] = rate
            store_samples[store_idx] = samples
            store_outcomes[store_idx] = 1.0 if rate >= 0.95 else 0.0
            
    factory_names = generate_unique_names(num_factories, "Factory")
    store_names = generate_unique_names(num_stores, "Store")
    
    return {
        'num_factories': num_factories,
        'num_stores': num_stores,
        'true_qualities': true_qualities,
        'is_failed_factory': is_failed_factory,
        'edges': edges,
        'store_outcomes': store_outcomes,
        'store_success_rates': store_success_rates,
        'store_samples': store_samples,
        'factory_names': factory_names,
        'store_names': store_names
    }

def generate_dataset(num_scenarios, seed=42):
    """
    Generates a list of random scenarios with varying parameters.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    dataset = []
    for _ in range(num_scenarios):
        num_factories = int(np.exp(np.random.uniform(np.log(5), np.log(500))))
        num_factories = max(5, min(500, num_factories))
        
        fail_rate = np.random.uniform(0.01, 0.20)
        num_fails = max(1, int(round(fail_rate * num_factories)))
        
        store_rate = np.random.uniform(0.20, 1.0)
        num_stores = max(1, int(round(store_rate * num_factories)))
        
        max_facs = min(15.0, float(num_factories))
        avg_factories_per_store = np.random.uniform(1.5, max_facs)
        
        sample_size = random.randint(50, 200)
        
        scenario = generate_company_scenario(
            num_factories=num_factories,
            num_fails=num_fails,
            num_stores=num_stores,
            avg_factories_per_store=avg_factories_per_store,
            sample_size=sample_size
        )
        dataset.append(scenario)
        
    return dataset

def scenario_to_report(scenario, complexity='a'):
    """
    Generates a natural language report based on complexity level:
    - 'a': template (spoonfeeds relevant information directly in sentences)
    - 'b': somewhat noisy (includes the relevant information but with varying sentence structures and some flowery language, model has to understand prose)
    - 'c': complex (could include updates on historical information, model has to recognize what the current information is)
    """
    num_stores = scenario['num_stores']
    edges = scenario['edges']
    store_outcomes = scenario['store_outcomes']
    store_success_rates = scenario['store_success_rates']
    store_names = scenario['store_names']
    factory_names = scenario['factory_names']
    
    store_supplies = {j: [] for j in range(num_stores)}
    for edge in edges:
        store_supplies[edge['store_idx']].append((edge['factory_idx'], edge['num_samples']))
        
    paragraphs = []
    
    intros_noisy = [
        "Store Quality Inspection Audit Report. Summary of recent shipments and sample quality tests.",
        "Weekly operations audit. This report details the performance of our retail outlets and their respective factory shipments.",
    ]
    
    outros_noisy = [
        "Please address any failures immediately to maintain company standards.",
        "Follow-up inspections will be scheduled for failed locations.",
    ]
    
    if complexity == 'a':
        # Level A: Template
        paragraphs.append("Template-based inspection report.")
        for edge in edges:
            fac_name = factory_names[edge['factory_idx']]
            store_name = store_names[edge['store_idx']]
            units = edge['num_samples']
            paragraphs.append(f"{fac_name} supplied {units} units to {store_name}.")
        for j in range(num_stores):
            store_name = store_names[j]
            outcome_str = "passed" if store_outcomes[j] == 1.0 else "failed"
            rate = store_success_rates[j]
            paragraphs.append(f"Inspection for {store_name} was {outcome_str} with success rate {rate:.1%}.")
        paragraphs.append("End of report.")
        
    elif complexity == 'b':
        # Level B: Somewhat Noisy Prose
        paragraphs.append(random.choice(intros_noisy))
        for j in range(num_stores):
            store_name = store_names[j]
            outcome_str = "passed the standard inspection" if store_outcomes[j] == 1.0 else "failed the quality audit"
            rate = store_success_rates[j]
            supplies = store_supplies[j]
            
            if not supplies:
                ship_narrative = "no deliveries were received during this window"
            else:
                parts = []
                for fac_idx, count in supplies:
                    fac_name = factory_names[fac_idx]
                    phrases = [
                        f"{count} units delivered by {fac_name}",
                        f"a shipment of {count} items from the {fac_name} location",
                        f"supplies consisting of {count} units from {fac_name}",
                    ]
                    parts.append(random.choice(phrases))
                if len(parts) == 1:
                    ship_narrative = parts[0]
                elif len(parts) == 2:
                    ship_narrative = f"{parts[0]} and {parts[1]}"
                else:
                    ship_narrative = ", ".join(parts[:-1]) + f", and {parts[-1]}"
                    
            narratives = [
                f"Regarding the outlet in {store_name}, it was noted that the location {outcome_str}. The final tested sample achieved a success rate of {rate:.1%}. Records show that they received {ship_narrative}.",
                f"Evaluation at {store_name} showed that the store {outcome_str} with {rate:.1%} success rate. This was after receiving {ship_narrative}.",
                f"The audit for {store_name} resulted in a success rate of {rate:.1%}, meaning they {outcome_str}. The log indicates {ship_narrative} was sent here.",
            ]
            paragraphs.append(random.choice(narratives))
        paragraphs.append(random.choice(outros_noisy))
        
    else: # complexity == 'c'
        # Level C: Complex (Historical updates, cancellations, re-inspections)
        paragraphs.append("Logistics and quality control operations update report.")
        
        # Process edges with historical updates
        for edge in edges:
            fac_name = factory_names[edge['factory_idx']]
            store_name = store_names[edge['store_idx']]
            units = edge['num_samples']
            
            r_val = random.random()
            if r_val < 0.3:
                # Cancelled shipment update
                cancelled_units = random.randint(30, 90)
                paragraphs.append(f"We originally planned for {fac_name} to supply {cancelled_units} units to {store_name}, but that delivery was cancelled. Instead, {fac_name} actually supplied {units} units to {store_name}.")
            elif r_val < 0.6:
                # Updated unit count update
                old_units = units + random.choice([-15, -10, 10, 15, 20])
                if old_units <= 0:
                    old_units = 50
                paragraphs.append(f"The shipment size from {fac_name} to {store_name} was initially set at {old_units} units. However, it was later updated, and the final quantity supplied was {units} units.")
            else:
                # Standard delivery sentence with a distractor cancellation
                other_fac = random.choice([f for f in factory_names if f != fac_name])
                cancelled_units = random.randint(10, 40)
                paragraphs.append(f"A scheduled delivery of {cancelled_units} units from {other_fac} to {store_name} was cancelled. Instead, {fac_name} shipped {units} units to the location.")
                
        # Generate re-inspection updates for stores
        for j in range(num_stores):
            store_name = store_names[j]
            outcome_str = "passed" if store_outcomes[j] == 1.0 else "failed"
            rate = store_success_rates[j]
            
            r_val = random.random()
            if r_val < 0.5:
                # Initial failure/different rate, then final rate
                old_rate = rate + random.choice([-0.1, -0.05, 0.05, 0.1])
                old_rate = max(0.50, min(0.99, old_rate))
                old_outcome = "passed" if old_rate >= 0.95 else "failed"
                paragraphs.append(f"The initial testing for {store_name} resulted in a success rate of {old_rate:.1%}, meaning it {old_outcome}. But after re-inspecting the stock, the final success rate was {rate:.1%}, and the location officially {outcome_str}.")
            else:
                # Re-inspection update
                paragraphs.append(f"For the {store_name} branch, auditing initially showed a rate of 90.0% (failed). After a subsequent second round inspection of the updated inventory, they reached {rate:.1%} success rate and were marked as {outcome_str}.")
                
        paragraphs.append("All operational updates have been logged.")
        
    return "\n\n".join(paragraphs)

def generate_phase_ii_dataset(num_scenarios, seed=42):
    """
    Generates a list of (scenario, report_text, complexity) tuples.
    """
    scenarios = generate_dataset(num_scenarios, seed=seed)
    random.seed(seed)
    
    dataset = []
    complexities = ['a', 'b', 'c']
    for idx, s in enumerate(scenarios):
        comp = complexities[idx % 3]
        report = scenario_to_report(s, complexity=comp)
        dataset.append((s, report, comp))
    return dataset

if __name__ == '__main__':
    test_scenarios = generate_phase_ii_dataset(3)
    for idx, (s, r, c) in enumerate(test_scenarios):
        print(f"\nScenario {idx} (Complexity: {c}):")
        print("-" * 50)
        print(r)
        print("-" * 50)
