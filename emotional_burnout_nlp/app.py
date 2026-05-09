from pathlib import Path
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Emotional Burnout Detector", page_icon="🧠", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/saved/bert_burnout"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please ensure the model is downloaded.")
        st.stop()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH), output_attentions=True)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

tokenizer, model = load_model()

st.title("🧠 Emotional Burnout Detection System")
st.markdown("**IT22196392 — Induwara K.P.Y. | R26-IT-059**")

st.warning(
    "⚠️ **Research Tool Only** — This system is not a substitute for professional mental health "
    "assessment. Predictions are AI-generated and may be inaccurate. If you or someone you know "
    "is in crisis, please contact a mental health professional or call a helpline immediately."
)
st.markdown("---")

text = st.text_area("Enter student text:", height=150,
                    placeholder="e.g. I feel so exhausted and hopeless about my studies...")

if st.button("Analyze", type="primary"):
    if not text or not text.strip():
        st.warning("Please enter some text!")
    elif len(text.strip().split()) < 3:
        st.warning("Please enter at least a few words for a meaningful prediction.")
    else:
        with st.spinner("Analyzing..."):
            try:
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

                if risk == 'Critical':
                    st.error("🚨 Critical risk detected. Please immediately connect this student with a mental health professional.")

                # Attention heatmap
                try:
                    all_attentions = torch.stack(outputs.attentions)  # [layers, batch, heads, seq, seq]
                    avg_attention = all_attentions.mean(dim=0).mean(dim=1).squeeze()  # avg layers & heads
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
                    plt.close(fig)
                except Exception:
                    st.info("Attention visualization unavailable for this input.")

                # All probabilities
                st.subheader("All Class Probabilities")
                for label, prob in zip(model.config.id2label.values(), probs):
                    st.progress(float(prob), text=f"{label}: {prob:.1%}")

            except Exception as e:
                st.error(f"Analysis failed: {e}. Please try again with different text.")
