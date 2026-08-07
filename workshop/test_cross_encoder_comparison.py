import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_models():
    models_to_test = [
        "cross-encoder/nli-deberta-v3-small",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ]
    
    premise1 = "Initially, Miami Summit Factory planned 100 units to Boston Store, but that delivery was cancelled. Instead, Seattle Onyx Factory actually delivered 80 units to Boston Store."
    
    pairs = [
        ("Miami Summit Factory", "Boston Store", False),  # Cancelled (Invalid)
        ("Seattle Onyx Factory", "Boston Store", True)    # Active delivery (Valid)
    ]
    
    for m_name in models_to_test:
        print(f"\n=================== Testing Model: {m_name} ===================")
        try:
            tokenizer = AutoTokenizer.from_pretrained(m_name)
            model = AutoModelForSequenceClassification.from_pretrained(m_name)
            model.eval()
            
            for fac, store, is_true in pairs:
                if "ms-marco" in m_name:
                    # Passage reranker: returns relevance score
                    inputs = tokenizer(f"{fac} supplied {store}", premise1, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        score = float(model(**inputs).logits[0][0])
                    print(f"  [{'VALID' if is_true else 'CANCELLED'}] Pair ({fac} -> {store}): Cross-Encoder Score = {score:.4f}")
                else:
                    # NLI Cross-Encoder
                    hypothesis = f"{fac} supplied {store}."
                    inputs = tokenizer(premise1, hypothesis, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].numpy()
                    print(f"  [{'VALID' if is_true else 'CANCELLED'}] Pair ({fac} -> {store}): Probs = [Contra: {probs[0]:.3f}, Neut: {probs[1]:.3f}, Entail: {probs[2]:.3f}]")
        except Exception as e:
            print(f"  Error testing {m_name}: {e}")

if __name__ == '__main__':
    test_models()
