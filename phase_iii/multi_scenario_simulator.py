import os
import sys
import re
import json
import torch
import argparse
import numpy as np

# Path setup to import phase_i, phase_ii, and phase_iii modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PHASE_I_DIR = os.path.join(BASE_DIR, 'phase_i')
PHASE_II_DIR = os.path.join(BASE_DIR, 'phase_ii')

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PHASE_I_DIR not in sys.path:
    sys.path.insert(0, PHASE_I_DIR)
if PHASE_II_DIR not in sys.path:
    sys.path.insert(0, PHASE_II_DIR)

from gnn_model import BipartiteGNN
from single_scenario_simulator import evaluate_prompt_text

def print_separator(char='-', length=80):
    print(char * length)

def extract_text_from_pdf(pdf_path):
    """
    Extracts plain text from a PDF file using available libraries (pypdf, fitz, pdfplumber, PyPDF2)
    or raw stream parsing fallback.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    text = ""
    # Try pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            return text
    except ImportError:
        pass

    # Try PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            return text
    except ImportError:
        pass

    # Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            t = page.get_text()
            if t:
                text += t + "\n"
        if text.strip():
            return text
    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text
    except ImportError:
        pass

    # Fallback to text file reading if passed a text file by mistake or raw text stream extraction
    try:
        with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            printable = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', content)
            if printable.strip():
                return printable
    except Exception:
        pass

    raise RuntimeError(f"Could not extract text from {pdf_path}. Please ensure a valid PDF reading library (pypdf, PyPDF2, PyMuPDF, or pdfplumber) is installed.")

def parse_prompts_pdf(raw_text):
    """
    Parses multiple scenario prompts from the extracted PDF text.
    Prompts can be demarcated by headers (e.g. 'Game 1', 'Scenario 1', 'Prompt 2').
    """
    chunks = re.split(r'(?i)(?=game\s+\d+|scenario\s+\d+|prompt\s+\d+|report\s+\d+)', raw_text)
    prompts = []
    for chunk in chunks:
        c = chunk.strip()
        if len(c) > 30:  # Valid scenario prompt block
            prompts.append(c)
            
    if not prompts:
        blocks = raw_text.split("\n\n")
        prompts = [b.strip() for b in blocks if len(b.strip()) > 30]
        
    return prompts

def parse_solutions_pdf(raw_text):
    """
    Parses ground truth solutions from the extracted solutions PDF text.
    Returns a list of dictionaries mapping factory names to ground truth statuses (1 for FAIL, 0 for PASS).
    """
    chunks = re.split(r'(?i)(?=game\s+\d+|scenario\s+\d+|solution\s+\d+|prompt\s+\d+)', raw_text)
    solutions = []
    
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        sol_map = {}
        # Match pattern "FactoryName: Rate" (e.g. "Alpha: 0.95" or "Bravo: 0.7") or explicit STATUS
        matches = re.findall(r'([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9\.]+)', c)
        for fac_raw, status_raw in matches:
            fac = fac_raw.strip().replace("Factory", "").strip()
            if not fac or fac.lower() in ['game', 'scenario', 'solution', 'prompt']:
                continue
            val_str = status_raw.strip().upper()
            if val_str in ['FAIL', 'FAILED', '1']:
                val = 1
            elif val_str in ['PASS', 'PASSED', '0']:
                val = 0
            else:
                try:
                    num_val = float(val_str)
                    val = 1 if num_val < 0.95 else 0
                except ValueError:
                    continue
            sol_map[fac] = val
        if sol_map:
            solutions.append(sol_map)
            
    return solutions

def evaluate_multiple_scenarios(prompts_pdf_path, solutions_pdf_path, output_dir=None):
    """
    Evaluates multiple scenario prompts from PDF files, computes predictions using the single_scenario_simulator,
    and calculates Accuracy, Precision, Recall, and F1-Score across all scenarios.
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR
        
    print_separator('=', 80)
    print("      MULTI-SCENARIO QUALITY CONTROL GNN EVALUATION SIMULATOR")
    print_separator('=', 80)
    
    print(f"[*] Reading prompts PDF: {prompts_pdf_path}")
    prompts_text = extract_text_from_pdf(prompts_pdf_path)
    prompts = parse_prompts_pdf(prompts_text)
    print(f"[*] Parsed {len(prompts)} scenario prompts.")
    
    print(f"[*] Reading solutions PDF: {solutions_pdf_path}")
    solutions_text = extract_text_from_pdf(solutions_pdf_path)
    solutions = parse_solutions_pdf(solutions_text)
    print(f"[*] Parsed {len(solutions)} ground truth scenario solutions.")
    
    # Load GNN model & threshold from Phase I
    model_path = os.path.join(PHASE_I_DIR, 'gnn_model.pth')
    device = torch.device('cpu')
    
    if not os.path.exists(model_path):
        print(f"[!] Warning: Trained GNN weights '{model_path}' not found. Using untrained GNN.")
        model = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
    else:
        print(f"[*] Loading frozen GNN model weights from '{model_path}'...")
        model = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    threshold_path = os.path.join(PHASE_I_DIR, 'thresholds.json')
    threshold = 0.25
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, 'r') as f:
                thresholds_dict = json.load(f)
                if 'GNN (Proposed)' in thresholds_dict:
                    threshold = thresholds_dict['GNN (Proposed)']
                    print(f"[*] Loaded validation-tuned GNN threshold from Phase I: {threshold:.2f}")
        except Exception:
            pass

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    all_factories = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet"]
    standard_stores = [
        "New York City", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
        "San Diego", "Dallas", "Seattle", "San Francisco", "Atlanta", "Boston", "Denver", 
        "Miami", "Detroit", "Memphis", "Nashville", "Raleigh", "Sacramento", "Springfield", 
        "Rochester", "Trenton", "Columbus", "Providence", "Peoria", "Hartford", "Buffalo", 
        "Casper", "Lincoln", "Pierre", "Topeka", "Laramie", "Fargo", "Biloxi", "Helena", 
        "Bismarck", "Yakima", "Salem", "Olympia", "Boise", "Bellingham", "Eugene", "Missoula", 
        "Spokane Valley", "Tacoma", "Portland"
    ]

    for idx, prompt_text in enumerate(prompts):
        print(f"\n>>> Running Evaluation for Scenario Prompt #{idx+1} <<<")
        
        gt_map = solutions[idx] if idx < len(solutions) else {}
        
        # Determine factory names from ground truth solution or prompt
        if gt_map:
            factory_names = list(gt_map.keys())
        else:
            factory_names = [f for f in all_factories if f in prompt_text]
            if not factory_names:
                factory_names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
                
        # Determine store names from prompt
        store_names = [s for s in standard_stores if re.search(r'\b' + re.escape(s) + r'\b', prompt_text, re.IGNORECASE)]
        if not store_names:
            store_names = ["New York City", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        
        est_qualities, fail_probs, pred_fails = evaluate_prompt_text(
            prompt_text,
            store_names=store_names,
            factory_names=factory_names,
            model=model,
            threshold=threshold,
            device=device
        )
        
        for f_idx, fac in enumerate(factory_names):
            pred_val = int(pred_fails[f_idx])
            prob_val = float(fail_probs[f_idx])
            true_val = gt_map.get(fac, 0)
            all_y_pred.append(pred_val)
            all_y_prob.append(prob_val)
            all_y_true.append(true_val)

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "=" * 80)
    print(" AGGREGATE EVALUATION METRICS ACROSS ALL SCENARIO PROMPTS")
    print("=" * 80)
    print(f"Total Prompts Evaluated : {len(prompts)}")
    print(f"Total Factory Predictions: {total}")
    print(f"Accuracy               : {accuracy:.2%}")
    print(f"Precision              : {precision:.2%}")
    print(f"Recall                 : {recall:.2%}")
    print(f"F1-Score               : {f1:.2%}")
    
    print("\nConfusion Matrix:")
    print(f"                      Predicted FAIL     Predicted PASS")
    print(f"True FAIL (Bad)      {tp:^14}     {fn:^14}")
    print(f"True PASS (Good)     {fp:^14}     {tn:^14}")
    print_separator('=', 80)
    
    # Save Plots
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Confusion Matrix Heatmap
        plt.subplot(1, 2, 1)
        cm = np.array([[tp, fn], [fp, tn]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted FAIL', 'Predicted PASS'],
                    yticklabels=['True FAIL', 'True PASS'])
        plt.title('Confusion Matrix across 10 Games')
        plt.ylabel('Ground Truth Status')
        plt.xlabel('GNN Prediction')
        
        # Subplot 2: Metrics Bar Chart
        plt.subplot(1, 2, 2)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_vals = [accuracy, precision, recall, f1]
        bars = plt.bar(metrics_names, metrics_vals, color=['#2b5c8f', '#4682b4', '#6baed6', '#9ecae1'])
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.title('GNN Quality Control Performance')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.1%}', ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'evaluation_metrics_plot.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[*] Saved evaluation plot artifact to '{plot_path}'")
    except Exception as e:
        print(f"[!] Could not generate plots: {e}")
    
    summary_path = os.path.join(output_dir, 'multi_scenario_evaluation_results.md')
    with open(summary_path, 'w') as f:
        f.write("# Multi-Scenario Evaluation Results\n\n")
        f.write(f"- **Total Prompts Evaluated**: {len(prompts)}\n")
        f.write(f"- **Total Factory Predictions**: {total}\n\n")
        f.write("### Aggregate Metrics\n")
        f.write(f"| Metric | Score |\n| --- | --- |\n")
        f.write(f"| **Accuracy** | {accuracy:.2%} |\n")
        f.write(f"| **Precision** | {precision:.2%} |\n")
        f.write(f"| **Recall** | {recall:.2%} |\n")
        f.write(f"| **F1-Score** | {f1:.2%} |\n\n")
        f.write("### Confusion Matrix\n")
        f.write(f"| | Predicted FAIL | Predicted PASS |\n| --- | --- | --- |\n")
        f.write(f"| **True FAIL** | {tp} | {fn} |\n")
        f.write(f"| **True PASS** | {fp} | {tn} |\n\n")
        f.write("![Evaluation Metrics Plot](evaluation_metrics_plot.png)\n")

    print(f"\n[*] Results saved to '{summary_path}'")
    return accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Multi-Scenario GNN Evaluation Simulator for PDF Prompts & Solutions")
    parser.add_argument('--prompts-pdf', type=str, required=True, help="Path to the PDF containing scenario prompts")
    parser.add_argument('--solutions-pdf', type=str, required=True, help="Path to the PDF containing scenario ground truth solutions")
    parser.add_argument('--output-dir', type=str, default=None, help="Directory to save evaluation markdown results")
    args = parser.parse_args()

    evaluate_multiple_scenarios(args.prompts_pdf, args.solutions_pdf, args.output_dir)

if __name__ == '__main__':
    main()
