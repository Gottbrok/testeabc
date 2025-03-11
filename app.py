import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
HF_CACHE = "/workspace/hf_cache"

# Configura cache do Hugging Face
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

# Carrega o modelo
@st.cache_resource
def load_model():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.rope_scaling = {"type": "yarn", "factor": 4.0}
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model()

# Interface
st.title("Qwen2.5-72B Code Analyzer")
uploaded_file = st.file_uploader("Upload de código/texto", type=["txt", "py", "md"])
user_input = st.text_area("Prompt:", height=200)
max_tokens = st.slider("Tamanho da resposta", 512, 8192, 2048)

if st.button("Enviar"):
    with st.spinner("Processando..."):
        # Combina conteúdo do arquivo + input
        file_content = uploaded_file.read().decode("utf-8") if uploaded_file else ""
        full_prompt = f"{file_content}\n{user_input}"
        
        # Gera resposta
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.code(response, language="python")
