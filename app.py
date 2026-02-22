import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import re
import random

# ==========================================
# ZONE 1: CONFIGURATION & STYLING
# (Matches: Preprocessing / Decision System)
# ==========================================
BLOCK_SIZE = 16 

def apply_custom_styles():
    st.set_page_config(page_title="History Architecture SLM", page_icon="📜")
    st.markdown("""
        <style>
        .stApp { background-color: #fdf5e6; }
        .history-card {
            background-color: #ffffff; padding: 25px; border-radius: 5px;
            border-left: 8px solid #8b4513; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            color: #2c1e14; font-family: 'Georgia', serif;
        }
        </style>
        """, unsafe_allow_html=True)

# ==========================================
# ZONE 2: PREPROCESSING LAYER
# (Matches: Preprocessing & Embedding)
# ==========================================
@st.cache_resource
def preprocessing_layer():
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    titles = re.findall(r"Event:\s*([\w\s-]+)\.", text)
    words = re.findall(r"[\w']+|[.,!?;:]", text)
    unique_words = sorted(list(set(words)))
    vocab_size = len(unique_words)
    wtoi = { w:i for i,w in enumerate(unique_words) }
    itow = { i:w for i,w in enumerate(unique_words) }
    data = torch.tensor([wtoi[w] for w in words], dtype=torch.long)
    
    return data, wtoi, itow, titles, vocab_size

# ==========================================
# ZONE 3: TRANSFORMER BLOCKS (THE BRAIN)
# (Matches: Embedding & Transformer Blocks)
# ==========================================
class HistoryEngine(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, 64)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, 64)
        self.lm_head = nn.Linear(64, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb
        logits = self.lm_head(x) 
        
        if targets is None:
            return logits, None
        
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / 0.2 # Fixed Temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# ZONE 4: DECISION & OUTPUT SYSTEM
# (Matches: Decision System & Output Layer)
# ==========================================
def training_loop(model, data, vocab_size):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(8000):
        ix = torch.randint(len(data) - BLOCK_SIZE, (32,))
        x = torch.stack([data[j:j+BLOCK_SIZE] for j in ix])
        y = torch.stack([data[j+1:j+BLOCK_SIZE+1] for j in ix])
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

# ==========================================
# ZONE 5: INPUT LAYER (MAIN EXECUTION)
# (Matches: Input Layer)
# ==========================================
def main():
    apply_custom_styles()
    st.title("📜 The History & Facts Archive")

    # Run Preprocessing
    data, wtoi, itow, titles, vocab_size = preprocessing_layer()

    # Run Transformer Training
    @st.cache_resource
    def get_trained_model():
        model = HistoryEngine(vocab_size)
        return training_loop(model, data, vocab_size)
    
    trained_model = get_trained_model()

    # UI Inputs
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("Enter a Year or Event:")
    with col2:
        length = st.slider("Detail Level:", 30, 120, 70)

    if st.button("🔍 Consult the Archive"):
        # Decision logic for seed
        if not search_query.strip():
            seed = ["Event", ":", random.choice(titles).split()[0]]
        else:
            seed = [search_query.strip().title()]

        if all(w in wtoi for w in seed):
            context = torch.tensor([[wtoi[w] for w in seed]], dtype=torch.long)
            gen_indices = trained_model.generate(context, max_new_tokens=length)
            tokens = [itow[i] for i in gen_indices[0].tolist()]
            
            # Post-processing the text
            result = " ".join(tokens).replace(" .", ".").replace(" ,", ",").replace(" :", ":")
            clean_fact = result.split("Event")[0] if search_query.strip() else result.split("Event")[1]

            st.markdown(f'<div class="history-card">Archive Record: {clean_fact}</div>', unsafe_allow_html=True)
        else:
            st.error("Archive not found.")

if __name__ == "__main__":
    main()