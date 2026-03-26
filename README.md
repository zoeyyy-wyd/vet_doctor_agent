# 🐶 Vet Triage Agent

A RAG-powered veterinary triage assistant that helps dog owners quickly assess the urgency of their pet's symptoms.

---

## What It Does

The agent accepts a video of the dog and/or a text description of symptoms, then returns a structured triage result with actionable advice. It pulls knowledge from authoritative veterinary sources (Merck Vet Manual, AVMA, PetMD, Cornell Canine Health Center, etc.) to ground its responses.

**Triage levels:**
- 🔴 **Red** — Urgent, seek emergency care immediately
- 🟡 **Yellow** — Concerning, visit a vet within 24 hours
- 🟢 **Green** — Mild, monitor at home

**Each result includes:** a plain-language summary, key risks, what to monitor, when to see a vet, a checklist to bring to the appointment, cited sources, and a disclaimer.

> ⚠️ This tool is for informational purposes only and does not replace professional veterinary diagnosis.

---

## How It Works

1. **Input** — User uploads a dog video and/or types a symptom description
2. **Video processing** — Frames are extracted and audio is transcribed via Whisper
3. **Case summary** — GPT-4o-mini fuses the visual and text inputs into a structured case description
4. **RAG retrieval** — Relevant passages are retrieved from a ChromaDB vector store built on veterinary websites
5. **Triage inference** — LLaMA generates a structured JSON triage result using the retrieved evidence
6. **UI** — Results are displayed in a Gradio web interface

---

## Evaluation

The project includes a dedicated evaluation notebook covering three dimensions:

- **MCQ Accuracy** — Tests whether the model selects the correct answer on veterinary multiple-choice questions, with and without RAG context
- **RAG Hit Rate** — Measures how often the retriever surfaces relevant documents
- **LLM-as-Judge Quality** — GPT scores each triage output on Safety, Completeness, and Tone on a 0–2 scale
  - *Safety*: Does the response avoid diagnosing or prescribing?
  - *Completeness*: Does it cover risks, monitoring tips, and vet guidance?
  - *Tone*: Is it calm, compassionate, and appropriately cautious?

Results are exported as CSVs and a six-panel visualization dashboard.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM (triage) | LLaMA |
| LLM (summary & eval) | GPT-4o-mini |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector store | ChromaDB |
| Speech-to-text | OpenAI Whisper |
| UI | Gradio |
| Runtime | Google Colab |
