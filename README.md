# AI Email-Drafter Agent (GPT-2 Medium + LoRA)

Author: Daksh Yadav
University: Indian Institute of Technology (IIT) Mandi
Department: Bioengineering
Date: September 15, 2025

------------------------------------------------------------
Project Summary
------------------------------------------------------------

The AI Email-Drafter Agent is a lightweight prototype that fine-tunes a transformer language model to generate polite, structured academic emails (e.g., extension requests, recommendation requests, leave requests, clarification emails).

To make training feasible on commodity GPUs, the project uses LoRA (PEFT) adapters on top of GPT-2 Medium (causal LM) so that only small adapter weights are trained and stored.

Goals:
- Produce email drafts with a subject, greeting, body and polite closing.
- Maintain a small storage footprint (LoRA adapters vs full model checkpoints).
- Provide deterministic post-processing to enforce structure and reduce failure modes.
- Log full interaction history (prompt -> raw output -> cleaned output) for reproducibility.

------------------------------------------------------------
Deliverables in this Repository
------------------------------------------------------------

Ai-Email-Agent/
├── demo/                     # screenshots + video demo
├── reports/
│   ├── architecture.pdf      # architecture document (components, flow, model choices)
│   ├── Data_Science_Report.pdf # data science report: fine-tuning setup + evaluation
│   └── interaction_logs.csv  # full eval interaction logs (prompt, raw, cleaned, flag)
├── Source Code - Python Notebook/  # Colab notebooks and supporting scripts
├── Project.ipynb - Colab.pdf # exported notebook PDF
├── train_example.jsonl       # sample training dataset
├── valid_example.jsonl       # sample validation / eval dataset
├── loss_curves.png           # training loss chart (example)
├── requirements.txt          # pip dependencies
└── README.txt                # this document

------------------------------------------------------------
Design Rationale
------------------------------------------------------------

- Base model: gpt2-medium (≈345M parameters). Chosen for good fluency while being runnable on a single 8–16 GB GPU with PEFT.
- PEFT / LoRA: training only small low-rank adapters -> low compute & storage cost.
- Tokenizer: GPT-2 BPE tokenizer + PAD token for batching.
- Post-processing: cleanup_and_validate() enforces Subject, Dear <Prof>, polite closing, and year normalization.
- UI: Gradio -> lightweight and easy demo deployment.

------------------------------------------------------------
Quick Start — Colab
------------------------------------------------------------
1. Open the main notebook in 'Source Code - Python Notebook/' or 'Project.ipynb' in Google Colab.
2. Switch runtime to GPU: Runtime -> Change runtime type -> GPU.
3. Install dependencies:
   !pip install -r requirements.txt
4. Run the notebook cells in order:
   - Data preparation
   - Fine-tuning (Trainer + LoRA)
   - Evaluation + log generation
   - Demo UI (Gradio)

------------------------------------------------------------
Quick Start — Local
------------------------------------------------------------
1. Clone repo:
   git clone https://github.com/<your-username>/Ai-Email-Agent.git
   cd Ai-Email-Agent

2. Setup environment:
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Run notebook or scripts for training/evaluation.

------------------------------------------------------------
Training
------------------------------------------------------------
Example training command:
python src/train_lora.py --train_file data/train.jsonl --validation_file data/valid.jsonl --output_dir lora-output --model_name_or_path gpt2-medium --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --num_train_epochs 8 --learning_rate 2e-4 --fp16

Or simply run the notebook cells.

------------------------------------------------------------
Evaluation & Interaction Logs
------------------------------------------------------------
1. Prepare eval.jsonl like:
   {"prompt": "Instruction: ...", "completion": "Subject: ..."}
2. Run evaluation script/notebook to produce reports/interaction_logs.csv.
3. File contains: prompt, reference_completion, raw_generation, cleaned_generation, valid_flag, settings.

------------------------------------------------------------
Demo UI (Gradio)
------------------------------------------------------------
Run:
python src/ui_gradio.py --adapter_dir lora-output --model_name gpt2-medium

Or use the Gradio cell in the Colab notebook.

------------------------------------------------------------
Results
------------------------------------------------------------
- Training loss: 0.0751
- Validation loss: 0.0693
- Validation perplexity: ~1.072
- Pass rate (structural correctness): ~85–90%

------------------------------------------------------------
Reports
------------------------------------------------------------
- architecture.pdf -> AI agent architecture, components, flow, model choices.
- Data_Science_Report.pdf -> dataset, preprocessing, fine-tuning setup, results, evaluation.
- interaction_logs.csv -> full evaluation logs.
