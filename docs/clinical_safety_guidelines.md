# Clinical Safety & Prompting Guidelines

## Overview

Medical information requires clear presentation and strict guardrails. This document outlines why we enforce strict formatting rules, ban markdown tables, and remove conversational AI filler.

---

## 1. Why We Banned Markdown Tables

When LLMs are asked to compare two things (like *Ibuprofen vs. Paracetamol*), their default instinct is to draw a markdown table:

```markdown
| Drug | Mechanism | Indications | Warnings |
|---|---|---|---|
| Ibuprofen | NSAID | Pain, Inflammation | GI bleeding |
| Paracetamol | Central | Pain, Fever | Liver toxicity |
```

### Why this is a bad idea in practice:
1. **Broken Mobile Viewports:** On a smartphone, markdown tables overflow horizontally. The text gets crammed into narrow 3-character-wide columns or forces the user to awkwardly scroll sideways inside a tiny chat bubble.
2. **Streaming Latency & Jitter:** Until the model reaches the end of a table row, the markdown parser doesn't know how to render the cells. This causes choppy, jumping text in the UI during streaming.
3. **Cramped Clinical Details:** Medical warnings (like renal impairment or cytochrome P450 interactions) cannot be properly explained in a 2-inch table cell.

### The Replacement: Section Headings with Bold Bullet Points
We prompt the model to use clear Markdown headings instead:

```markdown
### Overview
- **Ibuprofen:** Nonsteroidal anti-inflammatory drug (NSAID) that inhibits COX-1 and COX-2 enzymes.
- **Paracetamol (Acetaminophen):** Analgesic and antipyretic with central action and minimal anti-inflammatory effect.

### Clinical Indications
- **Ibuprofen:** Best for inflammatory pain (osteoarthritis, rheumatoid arthritis, dysmenorrhea, dental pain).
- **Paracetamol:** Best for non-inflammatory pain (headaches) and fever, especially when NSAIDs are contraindicated.

### Adverse Effects & Warnings
- **Ibuprofen:** Risk of gastric ulceration and bleeding with chronic use; renal strain.
- **Paracetamol:** Risk of dose-dependent hepatotoxicity (do not exceed 4g/day).
```

This renders cleanly on any screen size (mobile, tablet, desktop) without broken layouts.

---

## 2. Removing Conversational AI Filler

Most LLMs waste time with robotic pleasantries and meta-announcements:
- *"Certainly! I would be glad to help you compare these medications."*
- *"Based on the retrieved context from medical literature, here is the breakdown:"*
- *"As an AI language model, I want to emphasize..."*
- *"I hope this information was helpful to you! Feel free to ask if you have any further questions."*

### Why we stripped this:
In a clinical reference app, users want the facts immediately. Conversational filler wastes tokens, slows down time-to-first-token, and makes the application feel like a toy rather than a professional medical tool.

We enforce this directly in `src/prompt.py`:
- Start immediately with the first clinical heading (`### Overview`).
- Never use greetings or conversational openings.
- Conclude without polite sign-offs.

---

## 3. Strict Domain Guardrails

MedQuery is strictly a medical assistant. If a user asks:
- *"Write a Python script to scrape a website"*
- *"What is the exchange rate of USD to EUR?"*
- *"Who won the match yesterday?"*

The `TriageAgent` rejects it immediately with:
```text
I am a specialized medical assistant and can only answer questions related to health, 
medicine, and clinical topics. Please ask a health-related question.
```

No Pinecone vector search is triggered, no textbook context is loaded, and no tokens are wasted.
