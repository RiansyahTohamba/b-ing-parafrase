I'll create a Python program that paraphrases text and calculates the similarity score between the original and paraphrased versions.I've created a Python program that paraphrases text and calculates similarity scores! Here's what it does:

**Features:**
- **Paraphrasing**: Uses the Pegasus model to generate high-quality paraphrases
- **Dual Similarity Metrics**:
  - **Semantic Similarity**: Measures meaning similarity (0-1 scale)
  - **TF-IDF Similarity**: Measures word overlap and importance (0-1 scale)
- Generates multiple paraphrase variations

**Installation:**
```bash
pip install transformers torch sentence-transformers scikit-learn
```

**How to Use:**
1. Run the program as-is to see example output
2. Modify the `text` variable in `main()` with your own text
3. Adjust `num_paraphrases` to generate more or fewer variations

**Key Components:**
- `paraphrase()`: Generates paraphrased versions
- `calculate_semantic_similarity()`: Uses sentence embeddings for meaning comparison
- `calculate_tfidf_similarity()`: Uses word frequency for lexical comparison
- `process_text()`: Orchestrates paraphrasing and scoring

The similarity scores help you understand:
- **High scores (>0.8)**: Very similar meaning/words retained
- **Medium scores (0.5-0.8)**: Good paraphrase with variation
- **Low scores (<0.5)**: Significant rewording (might change meaning)

Would you like me to adjust the number of paraphrases, modify the similarity metrics, or add any other features?