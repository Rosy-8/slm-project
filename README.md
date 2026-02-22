## 📜 History Archive SLM (Small Language Model)

An AI-powered historical archive that uses a custom Word-Level Transformer to generate facts and details about significant historical events. Unlike large models that require internet access, this SLM trains locally on your machine in under 60 seconds.

## 🧠 Project Architecture
The project follows a modular AI pipeline, organized into five distinct layers:

**1.Input Layer:** A Streamlit-based interface for user queries.

**2.Preprocessing Layer:** Tokenizes raw text and maps words to numerical IDs.

**3.Embedding Layer:** Maps word IDs into a 64-dimensional vector space.

**4.Transformer Engine:** A lightweight neural network built with PyTorch.

**5.Output Layer:** Uses "Greedy Sampling" with a low temperature to ensure factual accuracy.

## 🚀 Getting Started

**Prerequisites**

. Python 3.8+

. Pip (Python package manager)

**Installation**

1.Clone the repository (or navigate to the folder):

cd History_SLM

2.Install dependencies:

pip install torch streamlit

3.Prepare the data:

Ensure your data.txt contains historical facts in the following format:

Event: Year Name. Fact: Description of the event.

## 🛠️ Usage
To run the application, execute the following command in your terminal:

python -m streamlit run app.py

**Features**
. Custom Search: Type a year (e.g., 1969) or a keyword (e.g., Titanic) to retrieve specific records.

. Detail Slider: Adjust the "Detail Level" to control the length of the generated narrative.

. Parchment UI: A custom-styled sepia interface designed for a historical experience.

## 📊 Technical Specs

. Framework: PyTorch (Neural Engine) & Streamlit (UI)

. Model Type: Word-Level Generative Transformer

. Memory (Block Size): 16 words

. Embedding Dimensions: 64

. Training Steps: 8,000 iterations

**⚖️ License**
This project is open-source and intended for educational purposes in the field of Small Language Modeling.
