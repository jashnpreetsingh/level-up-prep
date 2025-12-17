# 🌑 Training System

> *"Arise."* - Daily quest tracker for ML Engineer interview prep

A gamified learning tracker inspired by Solo Leveling, designed for systematic ML/DSA interview preparation. Track your progress, level up your skills, and become the Shadow Monarch of technical interviews.

## 🎯 What This Is

An open-source interview preparation system with:
- **DSA Schedule**: 4 weeks of structured LeetCode-style problems
- **ML/NLP Topics**: Comprehensive ML theory + NLP implementation focus
- **Math Foundations**: Linear Algebra, Probability, Optimization
- **Progress Tracking**: SQLite database to track completed problems, XP, and stats
- **Study Modal**: L5+ depth content with first-principles explanations

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start dev server (frontend + backend)
npm run dev:all

# Access at http://localhost:5173
```

## 📁 Project Structure

```
solo-leveling/
├── react-app/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── data/           
│   │   │   ├── content/     # ML/NLP/Math study content
│   │   │   ├── problemData.js
│   │   │   └── scheduleData.js
│   │   └── hooks/          # API hooks
│   ├── server/
│   │   ├── index.js        # Express API server
│   │   ├── database.js     # SQLite operations
│   │   └── shadow_monarch.db  # Progress database (tracked)
│   └── package.json
```

## 🎓 Syllabus Overview

### DSA
- Arrays, Linked Lists, Trees, Graphs
- Dynamic Programming, Binary Search
- Heaps, Tries, System Design
- Mock interviews & tapering

### Machine Learning (2 weeks)
- Core: Linear/Logistic Regression, SVM, Trees, Ensembles
- Deep Learning: Neural Networks, Transformers
- L5+ Systems: Recommendation, Distributed Training, Calibration

### NLP Coding (12 days)
- Tokenization, Embeddings, RNN/LSTM
- Attention, Transformers, Sequence Labeling
- LLM Internals (KV Cache, RoPE, Flash Attention)

### Math (4 weeks)
- Probability, Linear Algebra, Matrix Calculus
- Information Theory, Optimization

## 🗄️ Database

The `server/shadow_monarch.db` file is **intentionally tracked** in Git to:
- Share learning progress across devices
- Provide example data for open-source users
- Contains **only** problem completion status, notes, and stats (no sensitive data)

## 🛠️ Tech Stack

- **Frontend**: React + Vite
- **Backend**: Express.js
- **Database**: better-sqlite3
- **Styling**: Vanilla CSS (dark theme inspired by Solo Leveling)

## 📝 License

Open source for educational purposes. Feel free to fork and customize for your own interview prep!

## 🙏 Acknowledgments

Inspired by the Solo Leveling manhwa/anime. Built to help engineers level up their interview skills systematically.
