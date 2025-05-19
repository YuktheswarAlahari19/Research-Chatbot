import os
from transformers import AutoModelForCausalLM, AutoTokenizer  # For the language model
import torch  # For GPU usage
import nltk  # For text cleaning
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  # For finding themes
from sklearn.decomposition import LatentDirichletAllocation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


class ThemeIdentifier:
    
    def __init__(self, query_processor):
        
        self.query_processor = query_processor
        
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        max_memory = {0: "2GiB", "cpu": "16GiB"}
        device_map = {
            "model.embed_tokens": "cuda",
            "model.layers": "cuda",
            "model.norm": "cuda",
            "lm_head": "cpu"
        }
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                load_in_4bit=True,
                max_memory=max_memory,
                llm_int8_enable_fp32_cpu_offload=True
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Falling back to CPU")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                load_in_4bit=False
        )
        else:
            raise e
        
            
        self.stop_words = set(stopwords.words('english'))
        
        
    def clean_text(self, text):
        
        
        tokens = word_tokenize(text.lower())
        cleaned_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(cleaned_tokens)
    
    def find_themes(self, responses):
        
        texts = [response.get("answer", "") for response in responses]
        print(f"Texts for theme detection: {texts}")
        
        if not texts or all(t == "" for t in texts):
            return []
        
        
        
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        
        lda = LatentDirichletAllocation(n_components=2,max_iter=50,learning_decay=0.7, random_state=42)
        lda.fit(tfidf_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        
        themes = []
        for topic in lda.components_:
            top_words = [feature_names[i] for i in topic.argsort()[-3:]]  # Top 3 words
            themes.append(" ".join(top_words))
            
        print(f"Found themes: {themes}")
        return themes
    
    
    def summarize_responses(self, query: str, responses: list) -> dict:
        
        print(f"processing responses: {responses}")
      
        themes = self.find_themes(responses)
        
        
        if not responses:
            return {"summary": "No answers found.", "themes": [], "citations": []}
        
        
        answer_texts = [response.get("answer", "") for response in responses]
        citations = [response.get("citation", "Unknown") for response in responses]
        
        combined_text = " ".join(answer_texts)
        
        themes_str = ", ".join(themes) if themes else "none"
        citations_str = ", ".join(citations) if citations else "none"
        
        prompt = f"""
Your task is to:
- Summarize the provided answers in exactly one or two sentences, addressing the query '{query}'.
- Use a single period (.) to separate sentences in the summary if two are used.
- Do not add any extra text, explanations, commentary, or line breaks beyond the required summary.
- If no answers are provided, output only: No answers found.
- Replace [query] with the actual query provided.

Example:
Query: "What is machine learning?"
Answers: ["Machine learning uses algorithms to predict outcomes (Book A, p.10).", "It involves training models on data (Book B, p.25)."]

Output:
Machine learning uses algorithms to predict outcomes. It involves training models on data.

Answers: {combined_text}

Summary:
"""
        
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Summary:" in summary:
                summary = summary.split("Summary:")[1].strip()
                
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            summary = '. '.join(sentences[:2]) + '.' if sentences else "Could not summarize answers."
           
            if themes:
                summary += f" Themes: {themes_str}."
            if citations:
                summary += f" Citations: {citations_str}."
                
        except Exception as e:
            print(f"Error summarizing: {e}")
            summary = "Error summarizing answers."
            
            if themes:
                summary += f" Themes: {themes_str}."
            if citations:
                summary += f" Citations: {citations_str}."   

        return {
            "summary": summary,
            "themes": themes,
            "citations": citations
        }
        
        
        
    def identify_themes(self, query):
        responses = self.query_processor.process_query(query)
        print(f"Number of responses for '{query}': {len(responses)}")
        result = self.summarize_responses(query, responses)
        print(f"Final result: {result}")
        return result