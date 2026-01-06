import csv
import os
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict


# =========================
# CONFIGURATION
# =========================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CSV_FILE = "lexicon.csv"

# Load model and tokenizer once
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("Model loaded successfully!")


# =========================
# AI GENERATION RULES
# =========================

EXAMPLE_INPUT2 = """
Word: Supervised Learning
Category: Learning Types

Definition:
A type of machine learning where the model learns from labeled data, using input-output pairs to make predictions on unseen data.

Analogy:
Imagine teaching a child to recognize animals: 
- You show a dog and say "dog", a cat and say "cat". 
- The child learns and can later recognize new animals.

Example:
Training a system to tell if an email is spam by showing it many emails labeled as "spam" or "not spam."
"""

EXAMPLE_INPUT = """
(Reference example — do not copy content, only follow structure)

Word: Supervised Learning
Category: Learning Types

Definition:
It is a type of machine learning where the system learns using data that already has correct answers. The system uses this to make predictions on new data.

Analogy:
Like a teacher showing flashcards to a student.
Each card has a picture and the correct name written.
Over time, the student learns to recognize new pictures.

Example:
An email filter learning to detect spam by using emails marked as spam or not spam.
"""


SYSTEM_RULES = """
You are an educational content writer for MLera,
a learning platform for non-technical and first-year students.

Your task is to explain machine learning jargon in a way that is:
- Simple
- Friendly
- Non-academic
- Easy to understand without prior ML knowledge

STRICT RULES (must follow all):

GENERAL:
- Assume the reader is a first-year student with no technical background
- Do NOT use advanced ML jargon unless it is the word itself
- Do NOT sound like a textbook or research paper
- Do NOT repeat the jargon word unnecessarily

For a given machine learning jargon word, generate:

Definition:
- What it is, in very simple English
- Max 2 sentences
- If a formula or math symbol is needed, include it using LaTeX:
  - Use single dollar signs for inline math, e.g., "$x$"
  - Use double dollar signs for block math, e.g., "$$y = mx + c$$"

Analogy:
- Explain using daily life, or real-world situations with simple english
- Flesch Reading Ease score of atleast 75+.
- No technical or ML-related words
- Make it easy to imagine visually
- 2–4 short lines max

Example:
- Show how the concept is used in machine learning in real life
- Keep it practical and relatable
- No code
- Max 2 lines
"""


def build_prompt(jargon: str, category: str) -> str:
    return f"""
{SYSTEM_RULES}

Below is a reference example.
Follow only the FORMAT and STYLE, not the exact wording.

{EXAMPLE_INPUT}

Now generate content for the following word:

Word: {jargon}
Category: {category}

Return output strictly in this format:

Definition:
<text>

Analogy:
<text>

Example:
<text>
"""



# =========================
# MODEL GENERATION
# =========================

def generate_content(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": f"""Here is an example of the expected output format:

{EXAMPLE_INPUT}

Now generate content for the following:

{prompt}

Return output strictly in this format:

Definition:
<text>

Analogy:
<text>

Example:
<text>"""}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=500,
        temperature=0.6,
        do_sample=True,
        top_p=0.9
    )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    return generated_text


# =========================
# PARSER
# =========================

def parse_output(text: str) -> Dict[str, str]:
    sections = {"Definition": "", "Analogy": "", "Example": ""}
    
    for key in sections.keys():
        if key in text:
            try:
                parts = text.split(f"{key}:")
                if len(parts) > 1:
                    content = parts[1]
                    for other_key in sections.keys():
                        if other_key != key and other_key in content:
                            content = content.split(other_key)[0]
                    sections[key] = content.strip()
            except Exception as e:
                print(f"Error parsing {key}: {e}")
    
    return {
        "definition": sections["Definition"],
        "analogy": sections["Analogy"],
        "example": sections["Example"]
    }


# =========================
# CSV HANDLING
# =========================

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "jargon", "definition", "analogy", "example", "category"])


def append_to_csv(row: Dict[str, str]):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["id"],
            row["jargon"],
            row["definition"],
            row["analogy"],
            row["example"],
            row["category"]
        ])


# =========================
# MAIN PIPELINE
# =========================

def generate_lexicon_entry(jargon: str, category: str):
    prompt = f"Word: {jargon}\nCategory: {category}"
    raw_output = generate_content(prompt)
    parsed = parse_output(raw_output)
    
    row = {
        "id": str(uuid.uuid4()),
        "jargon": jargon,
        "definition": parsed["definition"],
        "analogy": parsed["analogy"],
        "example": parsed["example"],
        "category": category
    }
    
    append_to_csv(row)
    print(f"✅ Added: {jargon}")
    print(f"   Definition: {parsed['definition'][:50]}...")


# =========================
# EXAMPLE USAGE
# =========================

if __name__ == "__main__":
    init_csv()
    
    words = [
    # Core Concepts
    ("Data", "Core Concepts"),
    ("Dataset", "Core Concepts"),
    ("Feature", "Core Concepts"),
    ("Label", "Core Concepts"),
    ("Target", "Core Concepts"),
    ("Sample", "Core Concepts"),
    ("Model", "Core Concepts"),
    ("Algorithm", "Core Concepts"),
    ("Prediction", "Core Concepts"),
    ("Training", "Core Concepts"),

    # Learning Types
    ("Machine Learning", "Learning Types"),
    ("Supervised Learning", "Learning Types"),
    ("Unsupervised Learning", "Learning Types"),
    ("Semi-Supervised Learning", "Learning Types"),
    ("Reinforcement Learning", "Learning Types"),
    ("Transfer Learning", "Learning Types"),

    # Data Basics
    ("Dataset", "Data Basics"),
    ("Feature", "Data Basics"),
    ("Target", "Data Basics"),
    ("Noise", "Data Basics"),
    ("Structured Data", "Data Basics"),
    ("Unstructured Data", "Data Basics"),
    ("Tabular Data", "Data Basics"),
    ("Time Series", "Data Basics"),
    ("Missing Values", "Data Basics"),
    ("Outliers", "Data Basics"),
    ("Data Cleaning", "Data Basics"),
    ("Feature Selection", "Data Basics"),
    ("Data Preprocessing", "Data Basics"),
    ("Feature Engineering", "Data Basics"),
    ("Data Leakage", "Data Basics"),

    # Programming Basics
    ("Python", "Programming Basics"),
    ("Notebook", "Programming Basics"),
    ("Scikit Learn", "Programming Basics"),
    ("Environment", "Programming Basics"),
    ("Variable", "Programming Basics"),
    ("Function", "Programming Basics"),
    ("Loop", "Programming Basics"),
    ("Condition", "Programming Basics"),
    ("Library", "Programming Basics"),
    ("Package", "Programming Basics"),
    ("Script", "Programming Basics"),
    ("Debugging", "Programming Basics"),
    ("Version Control", "Programming Basics"),

    # Math Basics
    ("Mean", "Math Basics"),
    ("Median", "Math Basics"),
    ("Mode", "Math Basics"),
    ("Probability", "Math Basics"),
    ("Distribution", "Math Basics"),
    ("Variance", "Math Basics"),
    ("Standard Deviation", "Math Basics"),
    ("Vector", "Math Basics"),
    ("Matrix", "Math Basics"),

    # ML Models
    ("Linear Regression", "ML Models"),
    ("Logistic Regression", "ML Models"),
    ("Decision Tree", "ML Models"),
    ("Random Forest", "ML Models"),
    ("K-Nearest Neighbors", "ML Models"),
    ("Naive Bayes", "ML Models"),
    ("Support Vector Machine", "ML Models"),
    ("K-Means", "ML Models"),
    ("Principal Component Analysis", "ML Models"),
    ("Gradient Boosting", "ML Models"),
    ("XGBoost", "ML Models"),
    ("LightGBM", "ML Models"),
    ("CatBoost", "ML Models"),
    ("Hierarchical Clustering", "ML Models"),
    ("Convolutional Neural Network", "ML Models"),
    ("Principal Component Analysis", "ML Models"),
    ("Neural Network", "ML Models"),
    ("Deep Learning", "ML Models"),

    # Training Concepts
    ("Epoch", "Training Concepts"),
    ("Batch", "Training Concepts"),
    ("Learning Rate", "Training Concepts"),
    ("Loss", "Training Concepts"),
    ("Cost Function", "Training Concepts"),
    ("Optimization", "Training Concepts"),
    ("Gradient Descent", "Training Concepts"),
    ("Backpropagation", "Training Concepts"),
    ("Hyperparameter", "Training Concepts"),

    # Metrics
    ("Accuracy", "Metrics"),
    ("Precision", "Metrics"),
    ("Recall", "Metrics"),
    ("F1 Score", "Metrics"),
    ("True Positive", "Metrics"),
    ("False Positive", "Metrics"),
    ("True Negative", "Metrics"),
    ("False Negative", "Metrics"),
    ("Confusion Matrix", "Metrics"),
    ("Mean Squared Error", "Metrics"),
    ("Mean Absolute Error", "Metrics"),
    ("R-Squared", "Metrics"),

    # Bias Ethics
    ("Bias", "Bias Ethics"),
    ("Data Bias", "Bias Ethics"),
    ("Model Bias", "Bias Ethics"),
    ("Responsibile AI", "Bias Ethics"),
    ("Variance", "Bias Ethics"),
    ("Overfitting", "Bias Ethics"),
    ("Underfitting", "Bias Ethics"),
    ("Fairness", "Bias Ethics"),
    ("Explainability", "Bias Ethics"),
    ("Ethical AI", "Bias Ethics"),

    # AI Basics
    ("Artificial Intelligence", "AI Basics"),
    ("Generative AI", "AI Basics"),
    ("Large Language Model", "AI Basics"),
    ("Prompt", "AI Basics"),
    ("Token", "AI Basics"),
    ("Embedding", "AI Basics"),
    ("Context", "AI Basics"),
    ("Hallucination", "AI Basics"),
    ("Retrieval", "AI Basics"),
    ("Vector Database", "AI Basics"),
    ("Agentic AI", "AI Basics"),
    ("Automation", "AI Basics"),
    ("Agents", "AI Basics"),
    ("Chatbot", "AI Basics"),

    # Deployment Basics
    ("Deployment", "Deployment Basics"),
    ("Inference", "Deployment Basics"),
    ("API", "Deployment Basics"),
    ("Endpoint", "Deployment Basics"),
    ("Latency", "Deployment Basics"),
    ("Monitoring", "Deployment Basics"),
    ("Model Drift", "Deployment Basics"),
]

    
    for jargon, category in words:
        try:
            generate_lexicon_entry(jargon, category)
        except Exception as e:
            print(f"❌ Error processing {jargon}: {e}")
