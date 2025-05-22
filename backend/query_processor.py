# Importing libraries needed for processing queries and handling data
# transformers: Library for loading pre-trained language models and tokenizers
# torch: Library for machine learning tasks, used for tensor operations
# pandas: Library for handling data in tables (though not heavily used here)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd


# Defining a class called QueryProcessor to handle query processing using a language model

class QueryProcessor:
    
    # Constructor method that runs when a new QueryProcessor object is created
    # It takes a vector_store as input, which will be used to search for relevant documents
    # It also sets up a language model and tokenizer for answering queries
    
    def __init__(self, vector_store):
        
        self.vector_store = vector_store  # Storing the vector_store for later use
        model_name = "microsoft/Phi-3-mini-4k-instruct"  # Name of the pre-trained model to use
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Loading the tokenizer for the model
        self.model = AutoModelForCausalLM.from_pretrained(  # Loading the language model
            model_name,  # Model name
            torch_dtype=torch.float16,  # Using 16-bit precision to save memory
            device_map="auto",  # Automatically placing the model on GPU if available
            load_in_4bit=True  # Using 4-bit quantization to reduce memory usage
        )


    # Method to process a query and return answers based on search results
    # It takes a query (string) as input and returns a list of answers
    
    def process_query(self, query: str) -> list:
        answers = []  # Empty list to store the answers
        
        # Searching the vector store for documents matching the query, limiting to 2 results
        search_results = self.vector_store.search(query, max_results=2)
        
        # Printing the number of matching pages found for debugging
        print(f"Found {len(search_results)} matching pages:")
        # Looping through each search result to print the page number and a preview of the text
        for result in search_results:
            print(f"Page {result['page']}: {result['text'][:50]}...")


        # Looping through each search result to generate an answer
        
        for result in search_results:
            doc_id = result['doc_id']  # Getting the document ID from the search result
            page = result['page']  # Getting the page number from the search result
            text = result["text"][:700]  # Taking the first 700 characters of the text to keep input manageable
            
            # Creating a prompt to instruct the language model on how to answer the query
            prompt = f"""
Instructions: Answer '{query}' in one concise sentence using the text below.
Include '(Page {page})' at the end.
Output only the answer sentence, nothing else.

Text: {text}

Answer:
"""
            try:
                # Converting the prompt into a format the model can understand (tokenizing it)
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")  # Moving to GPU for faster processing
                input_length = len(inputs["input_ids"][0])  # Calculating the length of the tokenized input
                print(f"Input length for {doc_id}, page {page}: {input_length} tokens")  # Printing the input length for debugging

                # If the input is too long (more than 512 tokens), shorten the text and recreate the prompt
                if input_length > 512:
                    text = text[:500]  # Reducing the text to 500 characters
                    
                    prompt = f"""
Instructions: Answer '{query}' in one concise sentence using the text below.
Include '(Page {page})' at the end.
Output only the answer sentence, nothing else.

Text: {text}

Answer:
"""
                # Tokenizing the updated prompt (if shortened) and moving it to GPU
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                
                # Recalculating the input length after potentially shortening the text
                input_length = len(inputs["input_ids"][0])
                print(f"Input length for {doc_id}, page {page}: {input_length} tokens")  # Printing the new input length
                
                # Generating an answer using the language model, limiting to 100 new tokens
                outputs = self.model.generate(
                    **inputs, max_new_tokens=100
                )
                
                # Printing the type of the output for debugging
                print("DEBUG: Outputs type:", type(outputs))
                print("DEBUG: Outputs[0] type:", type(outputs[0]))
                
                
                # Decoding the model's output into readable text, skipping special tokens
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("DEBUG: Answer after decode:", answer, "Type:", type(answer))  # Printing the decoded answer
                
                
                # Checking if the answer contains the "Answer:" label and extracting the actual answer
                if "Answer:" in answer:
                    answer = answer.split("Answer:", 1)[1].strip()  # Taking the part after "Answer:"
                    print("DEBUG: Answer after split Answer:", answer, "Type:", type(answer))  # Printing the extracted answer
                    answer = answer.split('\n')[0].strip()  # Taking the first line and removing extra spaces
                    print("DEBUG: Answer after split newline:", answer, "Type:", type(answer))  # Printing the final answer
                    
                    
                else:
                    # If "Answer:" is not found, setting a default message
                    answer = "No answer found."
                    print("DEBUG: Answer set to default:", answer, "Type:", type(answer))
                
                # Ensuring the answer is a string before using string methods
                if not isinstance(answer, str):
                    print(f"WARNING: Answer is not a string for {doc_id}, page {page}: {answer}")
                    answer = str(answer)  # Converting to string if necessary
                    
                
                # Printing the cleaned answer for debugging
                print(f"Cleaned answer for {doc_id}, page {page}: {answer}")
                
                
                
                # Skipping answers that are invalid (empty, same as query, or contain "explanation of the process")
                if not answer or answer == query or "explanation of the process" in answer.lower():
                    print(f"Skipping invalid answer for {doc_id}, page {page}")
                    continue
                
                
                # Adding the valid answer to the list of answers
                answers.append({
                    "doc_id": doc_id,
                    "answer": answer,
                    "citation": f"Page {page}"
                })
            except Exception as e:
                # Printing any errors that occur during answer generation
                print(f"Error answering for {doc_id}, page {page}: {e}")
                
                
        
        # Returning the list of generated answers
        return answers


    # Method to display the answers in a readable format in the terminal
    # It takes a list of answers as input and prints them
    
    def display_answers(self, answers: list):
        
        # Checking if the answers list is empty
        if not answers:
            print("No answers found.")  # Printing a message if no answers are found
            return
        # Printing a header for the answers section
        print("\nAnswers to Your Question:")
        # Looping through each answer to print its details
        
        for answer in answers:
            print(f"Document: {answer['doc_id']}")  # Printing the document ID
            print(f"Answer: {answer['answer']}")  # Printing the generated answer
            print(f"Citation: {answer['citation']}")  # Printing the page citation
            print("---")  # Printing a separator line between answers
