from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

class QueryProcessor:
    
    def __init__(self, vector_store):
        self.vector_store = vector_store 
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
    def process_query(self, query: str) -> list:  # Changed return type annotation to list
        answers = []
        
        search_results = self.vector_store.search(query, max_results=2)
        
        # Print what pages we found to check if they’re relevant
        print(f"Found {len(search_results)} matching pages:")
        for result in search_results:
            print(f"Page {result['page']}: {result['text'][:50]}...")
        
        for result in search_results:
            doc_id = result['doc_id']
            page = result['page']
            text = result["text"][:700]
            
            prompt = f"""
Instructions: Answer '{query}' in one concise sentence using the text below.
Include '(Page {page})' at the end.
Output only the answer sentence, nothing else.

Text: {text}

Answer:
"""
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                input_length = len(inputs["input_ids"][0])
                print(f"Input length for {doc_id}, page {page}: {input_length} tokens")

                if input_length > 512:
                    text = text[:500]
                    prompt = f"""
Instructions: Answer '{query}' in one concise sentence using the text below.
Include '(Page {page})' at the end.
Output only the answer sentence, nothing else.

Text: {text}

Answer:
"""
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                
                input_length = len(inputs["input_ids"][0])
                print(f"Input length for {doc_id}, page {page}: {input_length} tokens")
                
                outputs = self.model.generate(
                    **inputs, max_new_tokens=100
                )
                
                print("DEBUG: Outputs type:", type(outputs))
                print("DEBUG: Outputs[0] type:", type(outputs[0]))
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("DEBUG: Answer after decode:", answer, "Type:", type(answer))
                
                if "Answer:" in answer:
                    answer = answer.split("Answer:", 1)[1].strip()
                    print("DEBUG: Answer after split Answer:", answer, "Type:", type(answer))
                    answer = answer.split('\n')[0].strip()
                    print("DEBUG: Answer after split newline:", answer, "Type:", type(answer))
                else:
                    answer = "No answer found."
                    print("DEBUG: Answer set to default:", answer, "Type:", type(answer))
                
                # Ensure answer is a string before calling lower()
                if not isinstance(answer, str):
                    print(f"WARNING: Answer is not a string for {doc_id}, page {page}: {answer}")
                    answer = str(answer)
                
                print(f"Cleaned answer for {doc_id}, page {page}: {answer}")
                
                if not answer or answer == query or "explanation of the process" in answer.lower():
                    print(f"Skipping invalid answer for {doc_id}, page {page}")
                    continue
                
                answers.append({
                    "doc_id": doc_id,
                    "answer": answer,
                    "citation": f"Page {page}"
                })
            except Exception as e:
                print(f"Error answering for {doc_id}, page {page}: {e}")
        
        return answers
    
    def display_answers(self, answers: list):
        if not answers:
            print("No answers found.")
            return
        print("\nAnswers to Your Question:")
        for answer in answers:
            print(f"Document: {answer['doc_id']}")
            print(f"Answer: {answer['answer']}")
            print(f"Citation: {answer['citation']}")
            print("---")