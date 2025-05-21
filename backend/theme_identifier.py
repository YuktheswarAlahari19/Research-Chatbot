import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk import pos_tag

# Set environment variable to optimize CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class ThemeIdentifier:
    def __init__(self, query_processor):
        """Initialize the ThemeIdentifier with a query processor and load the language model."""
        self.query_processor = query_processor
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Simplified device map for CPU/GPU compatibility
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

    def find_themes(self, query: str, responses: list) -> list:
        """Identify themes from responses based on the query."""
        # Ensure query is a string before calling lower()
        if isinstance(query, list):
            if query and isinstance(query[0], str):
                query = query[0]  # Extract the first string if the list contains one
            else:
                query = "default query"  # Fallback if the list is empty or doesn't contain a string
        elif not isinstance(query, str):
            query = str(query)  # Convert non-string, non-list inputs to string

        texts = [response.get("answer", "") for response in responses]
        if not texts or all(t == "" for t in texts):
            return [query.lower()]
        
        combined_text = " ".join(texts)
        prompt = f"""
Based on the query '{query}' and the text below, identify exactly 3 concise themes that capture the key concepts.
The first theme must be the main topic (e.g., '{query.lower()}').
The other two themes must be distinct, directly relevant to the text, non-repetitive, and use precise terms from the text.
Avoid vague words like 'figure' or 'workings'.
Return only the themes, separated by newlines.

Text: {combined_text}

Themes:
"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            themes_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Themes:" in themes_text:
                themes_text = themes_text.split("Themes:")[1].strip()
            else:
                themes_text = themes_text.strip()
            
            themes = [re.sub(r'^\d+\.\s*', '', theme).strip() for theme in themes_text.split('\n') if theme.strip()]
            
            if len(themes) >= 3:
                return themes[:3]
            elif len(themes) > 0:
                return themes + [query.lower()] * (3 - len(themes))
            
            tokens = word_tokenize(combined_text.lower())
            tagged = pos_tag(tokens)
            nouns = [word for word, pos in tagged if pos.startswith('NN') and word not in self.stop_words][:2]
            return [query.lower()] + nouns[:2] if nouns else [query.lower()] * 3
        
        except Exception as e:
            with open("theme_identifier_errors.log", "a") as f:
                f.write(f"Error generating themes: {e}\n")
            tokens = word_tokenize(combined_text.lower())
            tagged = pos_tag(tokens)
            nouns = [word for word, pos in tagged if pos.startswith('NN') and word not in self.stop_words][:2]
            return [query.lower()] + nouns[:2] if nouns else [query.lower()] * 3

    def summarize_responses(self, query: str, responses: list) -> dict:
        """Summarize responses and return structured output with themes and citations."""
        responses = self.query_processor.process_query(query)
        themes = self.find_themes(query, responses)
        
        if not responses:
            return {"summary": "No answers found.", "themes": [], "citations": [], "individual_responses": []}
        
        answer_texts = [response.get("answer", "") for response in responses]
        citations = [response.get("citation", "Unknown") for response in responses]
        doc_ids = [response.get("doc_id", "Unknown") for response in responses]
        combined_text = " ".join(answer_texts)
        
        prompt = f"""
Summarize the answers in 3 to 5 concise sentences for '{query}'.
Use a single period (.) between sentences.
Do not add extra text, explanations, or line breaks.
If no answers, output: No answers found.

Answers: {combined_text}

Summary:
"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=300)
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Summary:" in summary:
                summary = summary.split("Summary:")[1].strip()
            summary = re.split(r'\n\s*\n|What is|Summarize the answers', summary)[0].strip()
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            summary = '. '.join(sentences[:5]) + '.' if sentences else "Could not summarize answers."
        except Exception as e:
            with open("theme_identifier_errors.log", "a") as f:
                f.write(f"Error summarizing: {e}\n")
            summary = "Error summarizing answers."
        
        # Format the final summary
        final_summary = "Synthesized Answer:\n"
        final_summary += f"Summary: {summary}\n\n"
        final_summary += f"The query '{query}' is addressed through the following themes:\n"
        for i, theme in enumerate(themes, start=1):
            final_summary += f"- **Theme {i}: {theme}**\n"
        
        individual_responses = [
            {"Document ID": doc_id, "Extracted Answer": answer, "Citation": citation}
            for doc_id, answer, citation in zip(doc_ids, answer_texts, citations)
        ]
        
        return {
            "summary": final_summary,
            "themes": themes,
            "citations": citations,
            "individual_responses": individual_responses
        }

    def print_tabular_responses(self, responses):
        """Print individual responses in a clean tabular format."""
        print("\nIndividual Responses:")
        print("| Document ID                  | Extracted Answer                                      | Citation |")
        print("|-----------------------------|------------------------------------------------------|----------|")
        
        for resp in responses:
            doc_id = resp["Document ID"]
            answer = resp["Extracted Answer"][:50] + "..." if len(resp["Extracted Answer"]) > 50 else resp["Extracted Answer"]
            citation = resp["Citation"]
            print(f"| {doc_id:<27} | {answer:<52} | {citation:<8} |")

    def identify_themes(self, query: str) -> dict:
        """Process a query and return summarized results with themes."""
        responses = self.query_processor.process_query(query)
        result = self.summarize_responses(query, responses)
        return result