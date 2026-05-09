import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Emotional Burnout Detector", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('./models/saved/bert_burnout')
    model = AutoModelForSequenceClassification.from_pretrained(
        './models/saved/bert_burnout', output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Emotional Burnout Detection System")
st.markdown("**IT22196392 — Induwara K.P.Y. | R26-IT-059**")
st.markdown("---")

text = st.text_area("Enter student text:", height=150,
                    placeholder="e.g. I feel so exhausted and hopeless about my studies...")

if st.button("Analyze", type="primary"):
    if text:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)[0]
            predicted_id = probs.argmax().item()
            predicted_label = model.config.id2label[predicted_id]
            confidence = probs.max().item()
            
            risk_map = {'Normal': 'Low', 'Stress': 'Medium', 'Anxiety': 'Medium',
                       'Depression': 'High', 'Bipolar': 'High', 
                       'Personality disorder': 'High', 'Suicidal': 'Critical'}
            risk = risk_map.get(predicted_label, 'Unknown')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", predicted_label)
            col2.metric("Confidence", f"{confidence:.1%}")
            col3.metric("Risk Level", risk)
            
            # Attention heatmap
            attentions = outputs.attentions[-1]
            avg_attention = attentions.mean(dim=1).squeeze()
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            clean_tokens = tokens[1:-1]
            clean_attention = avg_attention[0].numpy()[1:-1]
            clean_attention = clean_attention / clean_attention.sum()
            
            fig, ax = plt.subplots(figsize=(12, 3))
            colors = plt.cm.Reds(clean_attention / clean_attention.max())
            ax.bar(range(len(clean_tokens)), clean_attention, color=colors)
            ax.set_xticks(range(len(clean_tokens)))
            ax.set_xticklabels(clean_tokens, rotation=45, ha='right')
            ax.set_title(f'Attention Heatmap — Words contributing to: {predicted_label}')
            plt.tight_layout()
            st.pyplot(fig)
            
            # All probabilities
            st.subheader("All Class Probabilities")
            for i, (label, prob) in enumerate(zip(model.config.id2label.values(), probs)):
                st.progress(float(prob), text=f"{label}: {prob:.1%}")
    else:
        st.warning("Please enter some text!")