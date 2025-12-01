"""
Text Paraphraser with Similarity Score Calculator

This program paraphrases input text and calculates similarity between
the original and paraphrased versions using multiple methods.

Requirements:
torch
    pip install transformers sentencepiece sentence-transformers scikit-learn
"""

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import torch

class TextParaphraser:
    def __init__(self):
        """Initialize paraphrasing and similarity models."""
        print("Loading models... This may take a moment on first run.")
        
        # Load paraphrasing model (Pegasus)
        model_name = "tuner007/pegasus_paraphrase"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        
        # Load sentence transformer for semantic similarity
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Models loaded successfully!\n")
    
    def paraphrase(self, text, num_return_sequences=1, max_length=60):
        """
        Paraphrase the input text.
        
        Args:
            text: Input text to paraphrase
            num_return_sequences: Number of paraphrases to generate
            max_length: Maximum length of paraphrased text
            
        Returns:
            List of paraphrased texts
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding="longest",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Generate paraphrase
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=10,
            num_return_sequences=num_return_sequences,
            temperature=1.5
        )
        
        # Decode outputs
        paraphrases = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return paraphrases
    
    def calculate_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity using sentence transformers.
        Returns score between 0 and 1 (higher = more similar).
        """
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()
    
    def calculate_tfidf_similarity(self, text1, text2):
        """
        Calculate TF-IDF cosine similarity.
        Returns score between 0 and 1 (higher = more similar).
        """
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return similarity
    
    def process_text(self, text, num_paraphrases=3):
        """
        Paraphrase text and calculate all similarity scores.
        
        Args:
            text: Input text to paraphrase
            num_paraphrases: Number of paraphrase variations to generate
            
        Returns:
            Dictionary with paraphrases and similarity scores
        """
        print(f"Original Text:\n{text}\n")
        print("=" * 60)
        
        # Generate paraphrases
        paraphrases = self.paraphrase(text, num_return_sequences=num_paraphrases)
        
        results = []
        for i, paraphrased in enumerate(paraphrases, 1):
            print(f"\nParaphrase {i}:")
            print(f"{paraphrased}\n")
            
            # Calculate similarities
            semantic_sim = self.calculate_semantic_similarity(text, paraphrased)
            tfidf_sim = self.calculate_tfidf_similarity(text, paraphrased)
            
            print(f"Semantic Similarity Score: {semantic_sim:.4f}")
            print(f"TF-IDF Similarity Score: {tfidf_sim:.4f}")
            print("-" * 60)
            
            results.append({
                'paraphrase': paraphrased,
                'semantic_similarity': semantic_sim,
                'tfidf_similarity': tfidf_sim
            })
        
        return results


def main():
    """Main function to run the paraphraser."""
    
    # Initialize paraphraser
    paraphraser = TextParaphraser()
    
    # Example text
    text = "Artificial intelligence is transforming the way we live and work in the modern world."
    
    # Process text
    results = paraphraser.process_text(text, num_paraphrases=3)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"\nParaphrase {i}: {result['paraphrase']}")
        print(f"  Semantic: {result['semantic_similarity']:.4f} | TF-IDF: {result['tfidf_similarity']:.4f}")
    
    print("\n" + "=" * 60)
    print("\nTo use with your own text:")
    print("1. Replace the 'text' variable with your input")
    print("2. Adjust num_paraphrases for more/fewer variations")
    print("\nExample:")
    print('  my_text = "Your text here"')
    print('  results = paraphraser.process_text(my_text)')


if __name__ == "__main__":
    main()