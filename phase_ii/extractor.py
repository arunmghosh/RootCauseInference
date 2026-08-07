import re
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np

def extract_relations_rules(report_text, store_names, factory_names):
    """
    Extracts connections between factories and stores using deterministic rules,
    handling historical updates, cancellations, and coreference.
    """
    store_set = {name.lower().strip(): name for name in store_names}
    factory_set = {name.lower().strip(): name for name in factory_names}
    
    sentences = re.split(r'(?<=[.!?])\s+', report_text)
    
    # Map (factory, store) -> {'units': int, 'cancelled': bool}
    connections_map = {}
    
    active_factory = None
    active_store = None
    
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue
        
        # Find mentioned factories and stores
        mentioned_factories = []
        for f_name in factory_names:
            if f_name.lower() in sentence_clean.lower():
                mentioned_factories.append(f_name)
        
        mentioned_stores = []
        for s_name in store_names:
            if s_name.lower() in sentence_clean.lower():
                mentioned_stores.append(s_name)
                
        # Update active context
        if mentioned_factories:
            active_factory = mentioned_factories[0]
        if mentioned_stores:
            active_store = mentioned_stores[0]
            
        # Parse numbers
        numbers = [int(n) for n in re.findall(r'\b\d+\b', sentence_clean) if int(n) < 1000]
        
        is_cancelled = "cancelled" in sentence_clean.lower()
        has_instead = "instead" in sentence_clean.lower()
        has_actually = "actually" in sentence_clean.lower() or "final" in sentence_clean.lower() or "updated to" in sentence_clean.lower()
        
        # 1. Cancelled plans with "instead" replacement (Level C)
        # E.g. "originally planned for F to supply U1 units to S, but that delivery was cancelled. Instead, F actually supplied U2 units to S."
        if is_cancelled and has_instead and len(mentioned_factories) >= 1 and len(mentioned_stores) >= 1 and len(numbers) >= 2:
            fac = mentioned_factories[0]
            store = mentioned_stores[0]
            connections_map[(fac, store)] = {'units': numbers[1], 'cancelled': False}
            continue
            
        # 2. Updated shipment size (Level C)
        # E.g. "shipment size from F to S was initially set at U1 units. However, it was later updated, and the final quantity supplied was U2 units."
        if "initially" in sentence_clean.lower() and has_actually and len(mentioned_factories) >= 1 and len(mentioned_stores) >= 1 and len(numbers) >= 2:
            fac = mentioned_factories[0]
            store = mentioned_stores[0]
            connections_map[(fac, store)] = {'units': numbers[1], 'cancelled': False}
            continue
            
        # 3. Simple cancellation (Level C)
        # E.g. "scheduled delivery of U1 units from F to S was cancelled."
        if is_cancelled and not has_instead and len(mentioned_factories) >= 1 and len(mentioned_stores) >= 1 and len(numbers) >= 1:
            fac = mentioned_factories[0]
            store = mentioned_stores[0]
            connections_map[(fac, store)] = {'units': numbers[0], 'cancelled': True}
            continue
            
        # 4. Coreference case: "It also supplied U units to S" (Level B)
        if not mentioned_factories and re.search(r'\bIt\s+(?:also\s+)?supplied\b', sentence_clean, re.IGNORECASE) and active_factory:
            for store in mentioned_stores:
                if len(numbers) >= 1:
                    connections_map[(active_factory, store)] = {'units': numbers[0], 'cancelled': False}
            continue
            
        # 5. Standard direct supply
        if not is_cancelled and len(mentioned_factories) >= 1:
            for fac in mentioned_factories:
                store = mentioned_stores[0] if mentioned_stores else active_store
                if store:
                    units = numbers[0] if numbers else 50
                    connections_map[(fac, store)] = {'units': units, 'cancelled': False}
            
    # Format connections list
    connections = []
    for (fac, store), info in connections_map.items():
        if not info['cancelled']:
            connections.append({
                'entity': fac,
                'event': store,
                'units': info['units']
            })
    return connections

def extract_store_outcomes_and_rates_rules(report_text, store_names):
    """
    Extracts the binary outcome and success rate for each store using deterministic rules,
    handling re-inspection updates (Level C).
    """
    store_info = {}
    sentences = re.split(r'(?<=[.!?])\s+', report_text)
    
    # Track multiple matches per store
    store_matches = {s: [] for s in store_names}
    active_store = None
    
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue
            
        mentioned_stores = []
        for s_name in store_names:
            if s_name.lower() in sentence_clean.lower():
                mentioned_stores.append(s_name)
                
        if mentioned_stores:
            active_store = mentioned_stores[0]
            
        target_stores = mentioned_stores if mentioned_stores else ([active_store] if active_store else [])
        
        outcome = None
        if 'passed' in sentence_clean.lower() or 'pass' in sentence_clean.lower() or 'success' in sentence_clean.lower() or 'free of defects' in sentence_clean.lower():
            if 'fail' not in sentence_clean.lower() or sentence_clean.lower().find('pass') > sentence_clean.lower().find('fail'):
                outcome = 'passed'
            else:
                outcome = 'failed'
        elif 'failed' in sentence_clean.lower() or 'fail' in sentence_clean.lower() or 'defect' in sentence_clean.lower():
            outcome = 'failed'
            
        rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', sentence_clean)
        rate = float(rate_match.group(1)) / 100.0 if rate_match else None
        
        if outcome or rate:
            for s_name in target_stores:
                store_matches[s_name].append({
                    'sentence': sentence_clean,
                    'outcome': outcome,
                    'rate': rate
                })
                    
    for s_name in store_names:
        matches = store_matches[s_name]
        if not matches:
            store_info[s_name] = {'outcome': 'passed', 'success_rate': 0.95}
            continue
            
        # Look for final or re-inspection updates
        final_match = None
        for m in matches:
            sent = m['sentence'].lower()
            if 'final' in sent or 're-inspect' in sent or 'second' in sent or 'officially' in sent:
                final_match = m
                break
                
        chosen = final_match if final_match else matches[-1]
        
        final_outcome = chosen['outcome'] if chosen['outcome'] else 'passed'
        final_rate = chosen['rate'] if chosen['rate'] is not None else 0.95
        
        store_info[s_name] = {
            'outcome': final_outcome,
            'success_rate': final_rate
        }
        
    return store_info

class RuleBasedExtractor:
    def extract(self, report_text, store_names, factory_names):
        """
        Rules-based extractor returning the 4-component schema.
        """
        # 1. Entities
        entities = []
        for f_name in factory_names:
            if f_name.lower() in report_text.lower():
                entities.append(f_name)
                
        # 2. Events
        events = []
        for s_name in store_names:
            if s_name.lower() in report_text.lower():
                events.append(s_name)
                
        # 3. Outcomes
        store_outcomes_rates = extract_store_outcomes_and_rates_rules(report_text, events)
        outcomes = {s_name: info['success_rate'] for s_name, info in store_outcomes_rates.items()}
        
        # 4. Connections
        connections = extract_relations_rules(report_text, events, entities)
        
        return {
            'entities': entities,
            'events': events,
            'outcomes': outcomes,
            'connections': connections
        }

class NLPExtractor:
    def __init__(self, device=None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[NLPExtractor] Loading dslim/bert-base-NER on {self.device}...")
        device_arg = -1
        if self.device == "cuda":
            device_arg = 0
        elif self.device == "mps":
            device_arg = "mps"
            
        try:
            self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=device_arg)
        except Exception:
            self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=-1)
            
        print(f"[NLPExtractor] Loading distilbert-base-cased-distilled-squad QA model...")
        qa_model_name = "distilbert-base-cased-distilled-squad"
        try:
            self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
            if self.device in ["cuda", "mps"]:
                try:
                    self.qa_model = self.qa_model.to(self.device)
                except Exception:
                    self.device = "cpu"
                    self.qa_model = self.qa_model.to("cpu")
        except Exception as e:
            print(f"[NLPExtractor] Warning loading QA model: {e}")
            self.qa_tokenizer = None
            self.qa_model = None

        print(f"[NLPExtractor] Loading Cross-Encoder Relation Classifier (cross-encoder/nli-deberta-v3-small)...")
        ce_model_name = "cross-encoder/nli-deberta-v3-small"
        try:
            self.ce_tokenizer = AutoTokenizer.from_pretrained(ce_model_name)
            from transformers import AutoModelForSequenceClassification
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(ce_model_name)
            if self.device in ["cuda", "mps"]:
                try:
                    self.ce_model = self.ce_model.to(self.device)
                except Exception:
                    self.ce_model = self.ce_model.to("cpu")
            self.ce_model.eval()
        except Exception as e:
            print(f"[NLPExtractor] Warning loading Cross-Encoder model: {e}")
            self.ce_tokenizer = None
            self.ce_model = None

        print("[NLPExtractor] NLP models loaded successfully.")

    def _verify_relation_cross_encoder(self, factory, store, text):
        """
        Solution C: Uses Cross-Encoder NLI to verify whether text entails an active supply relation
        or contradicts it (e.g. cancelled shipment).
        """
        if not self.ce_tokenizer or not self.ce_model or not text.strip():
            return True
        try:
            hypothesis = f"{factory} supplied products to {store}."
            inputs = self.ce_tokenizer(text, hypothesis, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.ce_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.ce_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            contradiction_prob = float(probs[0])
            if contradiction_prob > 0.4:
                return False
            return True
        except Exception:
            return True

    def _answer_question(self, question, context):
        if not self.qa_tokenizer or not self.qa_model or not context.strip():
            return ""
        try:
            inputs = self.qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.qa_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits)
            if end_idx >= start_idx:
                tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                return self.qa_tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:
            pass
        return ""

    def _merge_ner_entities(self, entities):
        merged = []
        current_entity = None
        for ent in entities:
            word = ent['word']
            entity_type = ent['entity']
            if word.startswith("##") and current_entity is not None:
                current_entity['word'] += word[2:]
            else:
                if current_entity is not None:
                    merged.append(current_entity)
                current_entity = {'word': word, 'entity': entity_type}
        if current_entity is not None:
            merged.append(current_entity)
            
        final_entities = []
        current = None
        for ent in merged:
            if current is None:
                current = ent
            else:
                if ent['entity'].split('-')[-1] == current['entity'].split('-')[-1]:
                    current['word'] += " " + ent['word']
                else:
                    final_entities.append(current)
                    current = ent
        if current is not None:
            final_entities.append(current)
            
        for ent in final_entities:
            ent['word'] = ent['word'].replace(" ##", "").replace("##", "").strip()
        return final_entities

    def _get_store_context(self, report_text, store_name):
        paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
        store_paras = [p for p in paragraphs if store_name.lower() in p.lower()]
        if store_paras:
            return "\n\n".join(store_paras)
            
        lines = [l.strip() for l in re.split(r'\n+|(?<=[.!?])\s+', report_text) if l.strip()]
        store_lines = [l for l in lines if store_name.lower() in l.lower()]
        if store_lines:
            return " ".join(store_lines)
        return report_text

    def extract(self, report_text, store_names, factory_names):
        # 1. Run BERT NER to extract entity candidates
        ner_results = self.ner_pipeline(report_text)
        entities_ner = self._merge_ner_entities(ner_results)
        
        detected_stores = set()
        detected_factories = set()
        
        for ent in entities_ner:
            ent_word = ent['word'].lower().strip()
            for s_name in store_names:
                if s_name.lower().strip() in ent_word or ent_word in s_name.lower().strip():
                    detected_stores.add(s_name)
            for f_name in factory_names:
                if f_name.lower().strip() in ent_word or ent_word in f_name.lower().strip():
                    detected_factories.add(f_name)
                    
        # Fallback to direct string search if NER missed some entities
        for s_name in store_names:
            if s_name.lower().strip() in report_text.lower():
                detected_stores.add(s_name)
        for f_name in factory_names:
            if f_name.lower().strip() in report_text.lower():
                detected_factories.add(f_name)

        detected_stores = list(detected_stores)
        detected_factories = list(detected_factories)
        
        # 2. Extract outcomes using targeted store context & QA / sentence updates
        target_store_list = store_names if store_names else detected_stores
        outcomes = {}
        for store in target_store_list:
            store_ctx = self._get_store_context(report_text, store)
            matches = list(re.finditer(r'(\d+(?:\.\d+)?)\s*%', store_ctx))
            if matches:
                # The last percentage in the store's context is the final reported inspection rate
                rate = float(matches[-1].group(1)) / 100.0
            else:
                ans_str = self._answer_question(f"What is the final inspection success rate of {store}?", store_ctx)
                qa_match = re.search(r'(\d+(?:\.\d+)?)\s*%', ans_str)
                if qa_match:
                    rate = float(qa_match.group(1)) / 100.0
                elif "passed" in store_ctx.lower() or "pass" in store_ctx.lower():
                    rate = 0.95
                elif "failed" in store_ctx.lower() or "fail" in store_ctx.lower():
                    rate = 0.90
                else:
                    fallback_info = extract_store_outcomes_and_rates_rules(report_text, [store])
                    rate = fallback_info.get(store, {}).get('success_rate', 0.95)
            outcomes[store] = rate
            
        # 3. Extract connections using dedicated Edge NLP Extractor (paragraph window & coreference resolution)
        connections = []
        paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
        target_factory_list = factory_names if factory_names else detected_factories
        
        for store in target_store_list:
            for factory in target_factory_list:
                store_fac_paras = [p for p in paragraphs if store.lower() in p.lower() and factory.lower() in p.lower()]
                if not store_fac_paras:
                    sentences = re.split(r'(?<=[.!?])\s+', report_text)
                    for i, sent in enumerate(sentences):
                        if store.lower() in sent.lower():
                            win = sentences[max(0, i-2):min(len(sentences), i+3)]
                            if any(factory.lower() in st.lower() for st in win):
                                store_fac_paras.append(" ".join(win))
                                break
                                
                for para in store_fac_paras:
                    # Filter out cancelled clauses if present
                    if "cancelled" in para.lower() or "cancel" in para.lower():
                        sents = re.split(r'(?<=[.!?])\s+', para)
                        actual_sent = [st for st in sents if factory.lower() in st.lower() and "cancel" not in st.lower()]
                        target_text = actual_sent[0] if actual_sent else para
                    else:
                        target_text = para
                        
                    # Solution C: Cross-Encoder Relation Verification
                    if not self._verify_relation_cross_encoder(factory, store, target_text):
                        continue
                        
                    ans_str = self._answer_question(f"How many units did {factory} actually supply to {store}?", target_text).lower()
                    if "cancel" in ans_str or "none" in ans_str or "zero" in ans_str:
                        continue
                        
                    m_num = re.search(r'\b\d+\b', ans_str)
                    units = int(m_num.group(0)) if m_num else None
                    if units is None:
                        m_units = re.search(r'\b(\d+)\s*units\b', target_text, re.IGNORECASE)
                        units = int(m_units.group(1)) if m_units else None
                    if units is None:
                        rb_conns = extract_relations_rules(target_text, [store], [factory])
                        if rb_conns:
                            units = rb_conns[0].get('units', 50)
                            
                    if units is not None and units > 0:
                        connections.append({
                            'entity': factory,
                            'event': store,
                            'units': units
                        })
                        break
                        
        return {
            'entities': detected_factories,
            'events': detected_stores,
            'outcomes': outcomes,
            'connections': connections
        }

def reconstruct_scenario(extracted_dict, true_scenario):
    """
    Reconstructs a scenario dictionary from the 4-component dictionary.
    """
    num_factories = true_scenario['num_factories']
    num_stores = true_scenario['num_stores']
    true_qualities = true_scenario['true_qualities']
    is_failed_factory = true_scenario['is_failed_factory']
    
    factory_names = true_scenario['factory_names']
    store_names = true_scenario['store_names']
    
    factory_to_idx = {name.lower().strip(): idx for idx, name in enumerate(factory_names)}
    store_to_idx = {name.lower().strip(): idx for idx, name in enumerate(store_names)}
    
    edges = []
    store_outcomes = np.zeros(num_stores, dtype=np.float32)
    store_success_rates = np.zeros(num_stores, dtype=np.float32)
    store_samples = np.zeros(num_stores, dtype=np.float32)
    
    # Defaults
    for j in range(num_stores):
        store_outcomes[j] = 1.0
        store_success_rates[j] = 0.95
        
    outcomes = extracted_dict.get('outcomes', {})
    for store_name_raw, rate_raw in outcomes.items():
        store_idx = -1
        if isinstance(store_name_raw, int) and store_name_raw < num_stores:
            store_idx = store_name_raw
        elif isinstance(store_name_raw, str):
            store_name_clean = store_name_raw.lower().strip()
            if store_name_clean in store_to_idx:
                store_idx = store_to_idx[store_name_clean]
                
        if store_idx != -1:
            rate = float(rate_raw)
            store_success_rates[store_idx] = rate
            store_outcomes[store_idx] = 1.0 if rate >= 0.95 else 0.0
            
    connections = extracted_dict.get('connections', [])
    store_units_sum = np.zeros(num_stores)
    
    for conn in connections:
        fac_name_raw = conn.get('entity', '')
        store_name_raw = conn.get('event', '')
        units = int(conn.get('units', 50))
        
        fac_idx = -1
        if isinstance(fac_name_raw, int) and fac_name_raw < num_factories:
            fac_idx = fac_name_raw
        elif isinstance(fac_name_raw, str):
            fac_name_clean = fac_name_raw.lower().strip()
            if fac_name_clean in factory_to_idx:
                fac_idx = factory_to_idx[fac_name_clean]
                
        store_idx = -1
        if isinstance(store_name_raw, int) and store_name_raw < num_stores:
            store_idx = store_name_raw
        elif isinstance(store_name_raw, str):
            store_name_clean = store_name_raw.lower().strip()
            if store_name_clean in store_to_idx:
                store_idx = store_to_idx[store_name_clean]
                
        if fac_idx != -1 and store_idx != -1:
            q = true_qualities[fac_idx]
            successes = np.random.binomial(units, q)
            
            edges.append({
                'factory_idx': fac_idx,
                'store_idx': store_idx,
                'num_samples': units,
                'num_successes': successes
            })
            store_units_sum[store_idx] += units
            
    for store_idx in range(num_stores):
        store_samples[store_idx] = store_units_sum[store_idx] if store_units_sum[store_idx] > 0 else 100.0
        
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

def compare_graphs(true_scenario, reconstructed_scenario):
    """
    Compares true scenario and reconstructed scenario.
    """
    true_outcomes = true_scenario['store_outcomes']
    recon_outcomes = reconstructed_scenario['store_outcomes']
    
    min_len = min(len(true_outcomes), len(recon_outcomes))
    outcome_accuracy = np.mean(true_outcomes[:min_len] == recon_outcomes[:min_len]) if min_len > 0 else 0.0
    
    true_edges = set((e['factory_idx'], e['store_idx']) for e in true_scenario['edges'])
    recon_edges = set((e['factory_idx'], e['store_idx']) for e in reconstructed_scenario['edges'])
    
    if not true_edges:
        edge_precision = 0.0
        edge_recall = 0.0
        edge_f1 = 0.0
    else:
        tp = len(true_edges.intersection(recon_edges))
        fp = len(recon_edges - true_edges)
        fn = len(true_edges - recon_edges)
        
        edge_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        edge_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0.0
        
    return {
        'outcome_accuracy': outcome_accuracy,
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'edge_f1': edge_f1,
        'num_true_edges': len(true_edges),
        'num_extracted_edges': len(recon_edges)
    }

if __name__ == '__main__':
    print("Testing extractor relation logic...")
    
    mock_s = {
        'num_factories': 2,
        'num_stores': 2,
        'true_qualities': np.array([0.98, 0.98]),
        'is_failed_factory': np.array([0, 0]),
        'store_outcomes': np.array([1.0, 1.0]),
        'store_success_rates': np.array([0.99, 0.99]),
        'store_samples': np.array([100, 100]),
        'factory_names': ["Seattle Onyx Factory", "Miami Summit Factory"],
        'store_names': ["Portland Jade Store", "Seattle Onyx Store"],
        'edges': [
            {'factory_idx': 0, 'store_idx': 0, 'num_samples': 41},
            {'factory_idx': 0, 'store_idx': 1, 'num_samples': 53}
        ]
    }
    
    report_b = (
        "Seattle Onyx Factory supplied 41 units to Portland Jade Store. It also supplied 53 units to Seattle Onyx Store.\n"
        "Inspection for Portland Jade Store passed with success rate 99.0%.\n"
        "Inspection for Seattle Onyx Store passed with success rate 99.0%."
    )
    
    rb = RuleBasedExtractor()
    extracted_rb = rb.extract(report_b, mock_s['store_names'], mock_s['factory_names'])
    print(json.dumps(extracted_rb, indent=2))
    
    recon = reconstruct_scenario(extracted_rb, mock_s)
    print("Reconstructed Edge Count:", len(recon['edges']))
    print("Edge Match:", compare_graphs(mock_s, recon))
