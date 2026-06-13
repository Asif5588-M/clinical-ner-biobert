import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd

st.set_page_config(
    page_title="Clinical NER — Medical Entity Extractor",
    page_icon="🏥",
    layout="wide"
)

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_ner():
    model_name = "asif-nawaz-ml/clinical-ner-biobert"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForTokenClassification.from_pretrained(model_name)
    ner        = pipeline(
        "ner",
        model          = model,
        tokenizer      = tokenizer,
        aggregation_strategy = "simple",
    )
    return ner

# ── Header ─────────────────────────────────────────────────
st.title("🏥 Clinical NER — Medical Entity Extractor")
st.caption("BioBERT fine-tuned | Extracts Drugs, Diseases & Pathogens | By Asif Nawaz, PMAS Arid Agriculture University")
st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Info")
    st.metric("Model", "BioBERT")
    st.metric("F1 Score", "93.56%")
    st.metric("Precision", "93.97%")
    st.metric("Recall", "94.59%")
    st.divider()

    st.header("🏷️ Entity Types")
    st.markdown("""
    - 💊 **Medicine** — Drug names
    - 🦠 **MedicalCondition** — Diseases & symptoms
    - 🧫 **Pathogen** — Viruses & bacteria
    """)
    st.divider()

    st.header("💡 Examples")
    examples = [
        "Patient has diabetes and was prescribed metformin 500mg.",
        "The patient tested positive for COVID-19 and was given azithromycin.",
        "Hypertension was managed with lisinopril and amlodipine.",
        "Tuberculosis caused by Mycobacterium tuberculosis was treated with isoniazid.",
        "Patient with pneumonia received amoxicillin and showed improvement.",
    ]
    for ex in examples:
        if st.button(ex[:50] + "...", use_container_width=True, key=ex):
            st.session_state["text"] = ex

# ── Main ───────────────────────────────────────────────────
text = st.text_area(
    "Enter clinical text:",
    value=st.session_state.get("text", ""),
    height=150,
    placeholder="e.g. Patient has diabetes and was prescribed metformin 500mg."
)

analyze = st.button("🔍 Extract Medical Entities", type="primary", use_container_width=True)

if analyze and text.strip():
    with st.spinner("Loading BioBERT model and analyzing..."):
        try:
            ner = load_ner()
            entities = ner(text)

            st.divider()

            if not entities:
                st.warning("No medical entities found.")
            else:
                # ── Highlighted text ───────────────────────────────
                st.subheader("📝 Highlighted Text")

                COLORS = {
                    "Medicine"        : "#FFD700",
                    "MedicalCondition": "#90EE90",
                    "Pathogen"        : "#FFB6C1",
                }

                highlighted = text
                # Sort by start position (reverse) to replace correctly
                sorted_ents = sorted(entities, key=lambda x: x["start"], reverse=True)
                for ent in sorted_ents:
                    word  = ent["word"]
                    label = ent["entity_group"]
                    color = COLORS.get(label, "#E0E0E0")
                    replacement = (
                        f'<mark style="background:{color};padding:2px 6px;'
                        f'border-radius:4px;font-weight:500">'
                        f'{word} <sup style="font-size:10px">{label}</sup></mark>'
                    )
                    highlighted = highlighted[:ent["start"]] + replacement + highlighted[ent["end"]:]

                st.markdown(highlighted, unsafe_allow_html=True)
                st.divider()

                # ── Entity table ───────────────────────────────────
                st.subheader(f"📊 Extracted Entities ({len(entities)})")

                df = pd.DataFrame([{
                    "Entity"    : e["word"],
                    "Type"      : e["entity_group"],
                    "Confidence": f"{e['score']:.2%}",
                    "Start"     : e["start"],
                    "End"       : e["end"],
                } for e in entities])

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.divider()

                # ── Summary ────────────────────────────────────────
                st.subheader("📈 Summary")
                col1, col2, col3 = st.columns(3)

                medicines   = [e for e in entities if e["entity_group"] == "Medicine"]
                conditions  = [e for e in entities if e["entity_group"] == "MedicalCondition"]
                pathogens   = [e for e in entities if e["entity_group"] == "Pathogen"]

                with col1:
                    st.metric("💊 Medicines", len(medicines))
                    for m in medicines:
                        st.write(f"• {m['word']}")

                with col2:
                    st.metric("🦠 Conditions", len(conditions))
                    for c in conditions:
                        st.write(f"• {c['word']}")

                with col3:
                    st.metric("🧫 Pathogens", len(pathogens))
                    for p in pathogens:
                        st.write(f"• {p['word']}")

                # ── Download ───────────────────────────────────────
                st.divider()
                st.download_button(
                    label     = "📥 Download Results (CSV)",
                    data      = df.to_csv(index=False),
                    file_name = "ner_results.csv",
                    mime      = "text/csv",
                    use_container_width = True,
                )

        except Exception as e:
            st.error(f"Error: {e}")

elif analyze and not text.strip():
    st.warning("Please enter clinical text!")

st.divider()
st.caption(
    "Model: BioBERT fine-tuned on clinical NER · "
    "F1: 93.56% · Asif Nawaz | PMAS Arid Agriculture University"
)