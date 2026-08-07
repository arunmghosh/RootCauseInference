import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CrossEncoderRelationClassifier:
    """
    Cross-Encoder NLI model for joint relation verification between (Factory, Store) pairs.
    Uses cross-encoder/nli-distilroberta-base (Encoder-only, fast inference).
    """
    def __init__(self, model_name="cross-encoder/nli-distilroberta-base", device="cpu"):
        self.device = device
        print(f"[CrossEncoder] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Labels for NLI: 0 = contradiction, 1 = neutral, 2 = entailment
        self.label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        
    def classify_pair(self, premise, factory, store):
        """
        Classifies whether premise entails that factory supplied products to store.
        """
        hypothesis = f"{factory} supplied products to {store}."
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            
        contradiction_prob = float(probs[0])
        neutral_prob = float(probs[1])
        entailment_prob = float(probs[2])
        
        return {
            'entailment': entailment_prob,
            'contradiction': contradiction_prob,
            'neutral': neutral_prob,
            'is_valid_relation': entailment_prob > contradiction_prob and entailment_prob > 0.35
        }

if __name__ == '__main__':
    ce = CrossEncoderRelationClassifier()
    
    # Test case 1: Active delivery
    premise1 = "Initially, Miami Summit Factory planned 100 units to Boston Store, but delivery was cancelled. Instead, Seattle Onyx Factory delivered 80 units to Boston Store."
    res1_a = ce.classify_pair(premise1, "Miami Summit Factory", "Boston Store")
    res1_b = ce.classify_pair(premise1, "Seattle Onyx Factory", "Boston Store")
    
    print("\n--- Test Case 1 Results ---")
    print("Miami Summit -> Boston Store:", res1_a)
    print("Seattle Onyx -> Boston Store:", res1_b)
