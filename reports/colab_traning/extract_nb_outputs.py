import json
import glob
from pathlib import Path

notebooks = glob.glob('/Users/ashwinshirke/Desktop/AIENG/OpenSLM/reports/colab_traning/*.ipynb')

for nb_file in notebooks:
    print(f"\n{'='*80}")
    print(f"Notebook: {Path(nb_file).name}")
    print(f"{'='*80}\n")
    
    with open(nb_file, 'r') as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError:
            print("Failed to decode JSON")
            continue
            
    for i, cell in enumerate(nb.get('cells', [])):
        # Look for code cells that might have output
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            if outputs:
                # Check if this cell looks like it ran a model generation
                source = "".join(cell.get('source', []))
                if 'generate' in source.lower() or 'print(' in source or 'sample' in source.lower() or '!make train' in source:
                    print(f"--- Cell {i} Source snippet: ---")
                    print(source[:200] + ("..." if len(source) > 200 else ""))
                    print("\n--- Outputs: ---")
                    for out in outputs:
                        if 'text' in out:
                            # stdout
                            text = "".join(out.get('text', []))
                            print("STDOUT:", text[:1000] + ("\n...[TRUNCATED]..." if len(text) > 1000 else ""))
                        elif 'data' in out and 'text/plain' in out['data']:
                            # execute_result
                            text = "".join(out['data']['text/plain'])
                            print("RESULT:", text[:1000] + ("\n...[TRUNCATED]..." if len(text) > 1000 else ""))
                        elif 'traceback' in out:
                            print("ERROR:", "".join(out['traceback']))
                    print("-" * 40 + "\n")
