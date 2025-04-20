# SeqGAN
Emotion-Driven Text Generation using SeqGAN :A full implementation of a text generation system using SeqGAN tailored for emotional expression. Combines supervised and reinforcement learning, custom reward functions (BLEU, emotion accuracy, diversity), and advanced sampling techniques (Top-k, Top-p, n-gram &amp; POS penalties).


# 🎭 Emotion-Driven Text Generation with SeqGAN

This repository contains the full implementation of a custom **SeqGAN-based system** for generating emotionally expressive text.
The project blends **supervised learning**, **reinforcement learning**, and **advanced sampling techniques** to teach a neural network how to write with emotional depth.

---

## 🧠 Project Overview

After building a classifier to identify emotions in text, this project takes the next step — generating sentences that reflect specific emotions (e.g., **fear**, **pride**, **embarrassment**).

### Model Architecture:
- **Generator**: LSTM with Bahdanau Attention.
- **Discriminator**: BiLSTM + CNN with dual-head outputs (Real/Fake + Emotion Classification).
- **Training Loop**: Pretraining (MLE), then adversarial training with Reinforcement Learning (RL).

### Reinforcement Learning:
Reward signal is a weighted combination of:
- **BLEU Score** – n-gram overlap with reference texts.
- **Emotion Accuracy** – using a pre-trained emotion classifier.
- **Discriminator Score** – coherence and realism.
- **Perplexity** – language fluency.
- **Repetition Penalty** – avoids loops and redundancy.
- **Saliency Score** – emphasizes meaningful content.


---

## 🧪 Sampling Strategy

The generator uses a robust sampling system including:
- **Temperature Sampling**
- **Top-k / Top-p Filtering**
- **n-gram Penalties (Jaccard + repetition)**
- **POS-based Penalties** (e.g., repeated verbs)
- **Content Constraints** – avoids functional word loops and ensures informative tokens

> These manipulations increase diversity, coherence, and emotional alignment.


---

## 📊 Visualization & Analysis

All training stages are logged and visualized:
- Generator & Discriminator pretraining
- RL loss and reward dynamics
- Distinct-n diversity metrics
- Emotion-specific output quality by temperature

👉 A full analysis of the results is included in the accompanying PDF.


---

## 🔗 Try It Out

You can explore the results and generated texts by running the final sampling function, specifying the desired emotion. Example:
```python
samples = generate_from_generator(generator, word_to_index, index_to_word, num_samples=5, temperature=0.9, mode="temperature")
```

---

## 📁 Files
- `generator.py` – Model + training logic
- `discriminator.py` – Dual-head classifier
- `sampling.py` – Smart sampling logic
- `rewards.py` – Reward function components
- `train.py` – Full training loop
- `visuals/` – Contains graphs and analysis

---

## 🤝 Contributing
Pull requests and suggestions are welcome!
If you find the project useful, give it a ⭐ and share your thoughts.

---

## 📫 Contact
Feel free to reach out via LinkedIn if you want to collaborate or just geek out over GANs and emotions.

---

## 🧠 Inspiration
This project was inspired by the challenge of not only understanding emotions — but teaching a machine to **express** them.
