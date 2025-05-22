# Importing libraries needed for theme identification and natural language processing
# os: Library for interacting with the operating system, used to set environment variables
# transformers: Library for loading pre-trained language models and tokenizers
# torch: Library for machine learning tasks, used for tensor operations
# nltk: Library for natural language processing tasks like tokenization and stop words
# re: Library for working with regular expressions to clean text


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk import pos_tag


# Setting an environment variable to optimize memory usage for PyTorch on CUDA (GPU)
# This helps prevent memory issues when using the GPU for processing


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Downloading required NLTK data for text processing
# punkt: For tokenizing text into words
# punkt_tab: Additional data for tokenization
# stopwords: For removing common words like "the", "is", etc.
# averaged_perceptron_tagger_eng: For tagging parts of speech in English text
# quiet=True: Prevents NLTK from printing download messages


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# Defining a class called ThemeIdentifier to identify themes in responses to a query
class ThemeIdentifier:
    # Constructor method that runs when a new ThemeIdentifier object is created
    # It takes a query_processor as input and sets up a language model for theme identification
    
    
    def __init__(self, query_processor):
      
        self.query_processor = query_processor  # Storing the query processor for later use
        model_name = "microsoft/Phi-3-mini-4k-instruct"  # Name of the pre-trained model to use
        
        # Loading the tokenizer for the model, which converts text into a format the model can understand
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setting up a memory limit for the model to prevent memory issues
        # 2GiB for GPU (device 0) and 16GiB for CPU
        max_memory = {0: "2GiB", "cpu": "16GiB"}
        # Defining how the model parts should be distributed between GPU and CPU
        device_map = {
            "model.embed_tokens": "cuda",  # Embedding tokens on GPU
            "model.layers": "cuda",  # Model layers on GPU
            "model.norm": "cuda",  # Normalization on GPU
            "lm_head": "cpu"  # Language model head on CPU
        }
        
        try:
            # Loading the language model with the specified settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,  # Model name
                torch_dtype=torch.float16,  # Using 16-bit precision to save memory
                device_map=device_map,  # Distributing model parts between GPU and CPU
                load_in_4bit=True,  # Using 4-bit quantization to reduce memory usage
                max_memory=max_memory,  # Setting memory limits
                llm_int8_enable_fp32_cpu_offload=True  # Enabling CPU offloading for 8-bit operations
            )
        except RuntimeError as e:
            # Handling memory errors by falling back to CPU if GPU runs out of memory
            if "out of memory" in str(e):
                print("Falling back to CPU")  # Informing the user that we're switching to CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,  # Model name
                    torch_dtype=torch.float16,  # Using 16-bit precision
                    device_map="cpu",  # Placing the entire model on CPU
                    load_in_4bit=False  # Disabling 4-bit quantization
                )
            else:
                # Raising any other errors that occur during model loading
                raise e
        
        # Creating a set of common English words (stop words) to ignore during theme identification
        self.stop_words = set(stopwords.words('english'))


    # Method to identify themes from responses based on a query
    # It takes a query (string) and a list of responses as input, and returns a list of themes
    
    
    def find_themes(self, query: str, responses: list) -> list:
       
        # Ensuring the query is a string before using string methods like lower()
        if isinstance(query, list):  # Checking if the query is a list
            if query and isinstance(query[0], str):  # If the list has a string as its first element
                query = query[0]  # Extracting the first string
            else:
                query = "default query"  # Using a default query if the list is empty or not a string
        elif not isinstance(query, str):  # If the query is not a string or list
            query = str(query)  # Converting it to a string

        # Extracting the answer texts from the responses, defaulting to an empty string if not found
        texts = [response.get("answer", "") for response in responses]
        # Checking if there are no texts or all texts are empty
        if not texts or all(t == "" for t in texts):
            return [query.lower()]  # Returning the query in lowercase as a fallback
        
        # Combining all answer texts into a single string for processing
        combined_text = " ".join(texts)
        # Creating a prompt to instruct the language model to identify themes
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
            # Converting the prompt into a format the model can understand (tokenizing it)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Generating themes using the language model, limiting to 100 new tokens
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            # Decoding the model's output into readable text, skipping special tokens
            themes_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extracting the themes part from the output
            
            if "Themes:" in themes_text:
                themes_text = themes_text.split("Themes:")[1].strip()  # Taking the part after "Themes:"
            else:
                themes_text = themes_text.strip()  # Removing extra spaces
            
            # Cleaning the themes by removing any numbering (e.g., "1. ") and extra spaces
            themes = [re.sub(r'^\d+\.\s*', '', theme).strip() for theme in themes_text.split('\n') if theme.strip()]
            
            # Ensuring exactly 3 themes are returned
            
            if len(themes) >= 3:  # If we have 3 or more themes
                return themes[:3]  # Returning the first 3 themes
            elif len(themes) > 0:  # If we have 1 or 2 themes
                return themes + [query.lower()] * (3 - len(themes))  # Filling the rest with the query in lowercase
            
        
            # Fallback method if the model fails: extracting nouns from the text
            
            tokens = word_tokenize(combined_text.lower())  # Breaking the text into words
            tagged = pos_tag(tokens)  # Tagging each word with its part of speech (e.g., noun, verb)
            
            # Extracting nouns (words starting with 'NN') that are not stop words, limiting to 2
            nouns = [word for word, pos in tagged if pos.startswith('NN') and word not in self.stop_words][:2]
            
            # Returning the query plus the extracted nouns, or just the query if no nouns are found
            return [query.lower()] + nouns[:2] if nouns else [query.lower()] * 3
        
        except Exception as e:
            # Logging any errors that occur during theme generation to a file
            with open("theme_identifier_errors.log", "a") as f:
                f.write(f"Error generating themes: {e}\n")
                
            # Using the same fallback method as above if the model fails
            
            tokens = word_tokenize(combined_text.lower())
            tagged = pos_tag(tokens)
            nouns = [word for word, pos in tagged if pos.startswith('NN') and word not in self.stop_words][:2]
            return [query.lower()] + nouns[:2] if nouns else [query.lower()] * 3


    # Method to summarize responses and return a structured output with themes and citations
    # It takes a query (string) and a list of responses as input, and returns a dictionary
    
    
    def summarize_responses(self, query: str, responses: list) -> dict:
        
        # Processing the query to get new responses using the query processor
        responses = self.query_processor.process_query(query)
        # Identifying themes from the responses using the find_themes method
        themes = self.find_themes(query, responses)
        
        # Checking if there are no responses
        if not responses:
            # Returning an empty result if no responses are found
            return {"summary": "No answers found.", "themes": [], "citations": [], "individual_responses": []}
        
        # Extracting the answer texts, citations, and document IDs from the responses
        answer_texts = [response.get("answer", "") for response in responses]
        citations = [response.get("citation", "Unknown") for response in responses]
        doc_ids = [response.get("doc_id", "Unknown") for response in responses]
        # Combining all answer texts into a single string for summarization
        combined_text = " ".join(answer_texts)
        
        # Creating a prompt to instruct the language model to summarize the answers
        prompt = f"""
Summarize the answers in 3 to 5 concise sentences for '{query}'.
Use a single period (.) between sentences.
Do not add extra text, explanations, or line breaks.
If no answers, output: No answers found.

Answers: {combined_text}

Summary:
"""
        try:
            # Converting the prompt into a format the model can understand (tokenizing it)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generating a summary using the language model, limiting to 300 new tokens
            outputs = self.model.generate(**inputs, max_new_tokens=300)
            
            # Decoding the model's output into readable text, skipping special tokens
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extracting the summary part from the output
            if "Summary:" in summary:
                summary = summary.split("Summary:")[1].strip()  # Taking the part after "Summary:"
                
            # Removing unwanted text and splitting into sentences
            summary = re.split(r'\n\s*\n|What is|Summarize the answers', summary)[0].strip()
            
            # Cleaning the sentences by removing extra spaces
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            
            # Joining the sentences back together with periods, limiting to 5 sentences
            summary = '. '.join(sentences[:5]) + '.' if sentences else "Could not summarize answers."
            
        except Exception as e:
            # Logging any errors that occur during summarization to a file
            with open("theme_identifier_errors.log", "a") as f:
                f.write(f"Error summarizing: {e}\n")
            # Setting a default message if summarization fails
            summary = "Error summarizing answers."
        
        # Formatting the final summary string with the synthesized answer and themes
        
        
        final_summary = "Synthesized Answer:\n"
        final_summary += f"Summary: {summary}\n\n"
        final_summary += f"The query '{query}' is addressed through the following themes:\n"
        # Adding each theme with a number and bold formatting
        for i, theme in enumerate(themes, start=1):
            final_summary += f"- **Theme {i}: {theme}**\n"
        
        # Creating a list of individual responses with document ID, answer, and citation
        individual_responses = [
            {"Document ID": doc_id, "Extracted Answer": answer, "Citation": citation}
            for doc_id, answer, citation in zip(doc_ids, answer_texts, citations)
        ]
        
        # Returning a dictionary with the summary, themes, citations, and individual responses
        return {
            "summary": final_summary,
            "themes": themes,
            "citations": citations,
            "individual_responses": individual_responses
        }


    # Method to print individual responses in a clean table format in the terminal
    # It takes a list of responses as input and prints them
    
    
    def print_tabular_responses(self, responses):
      
        # Printing a header for the responses section
        print("\nIndividual Responses:")
        # Printing the table header with column names
        print("| Document ID                  | Extracted Answer                                      | Citation |")
        # Printing a separator line for the table header
        print("|-----------------------------|------------------------------------------------------|----------|")
        
        # Looping through each response to print its details in the table
        
        for resp in responses:
            doc_id = resp["Document ID"]  # Getting the document ID
            # Truncating the answer to 50 characters for display, adding "..." if longer
            answer = resp["Extracted Answer"][:50] + "..." if len(resp["Extracted Answer"]) > 50 else resp["Extracted Answer"]
            citation = resp["Citation"]  # Getting the citation
            # Printing the row with formatted spacing
            print(f"| {doc_id:<27} | {answer:<52} | {citation:<8} |")


    # Method to process a query and return summarized results with themes
    # It takes a query (string) as input and returns a dictionary
    
    def identify_themes(self, query: str) -> dict:
       
        # Processing the query to get responses
        responses = self.query_processor.process_query(query)
        # Summarizing the responses and identifying themes
        result = self.summarize_responses(query, responses)
        # Returning the summarized result
        return result
