import fitz  # PyMuPDF
import multiprocessing
from pydantic import BaseModel, Field
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from langchain.text_splitter import RecursiveCharacterTextSplitter
import instructor

class GenerateQuestion(BaseModel):
    question1: str = Field(description="A question related to document")
    question2: str = Field(description="A question related to document")
    question3: str = Field(description="A very short question without a clear subject or object.")
    question4: str = Field(description="A very short question without a clear subject or object.")
    question5: str = Field(description="A general question related to the overall topic or concept.")
    question6: str = Field(description="A general question related to the overall topic or concept.")

def read_pdf(file_path):
    """Extract text from PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=5000, chunk_overlap=1000):
    """Chunk the text using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def initialize():
        
    llama = Llama(
        model_path='/Users/thanhnguyen/Documents/Developer/multimodel_rag_pdf/models/llm/SanctumAI-meta-llama-3-8b-instruct.Q8_0.gguf',
        n_ctx=18000,
        n_gpu_layers=20,
        n_batch=850,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_threads=multiprocessing.cpu_count() - 3,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),  # (1)!
        logits_all=True,
        verbose=True,
    )

    create = instructor.patch(
        create=llama.create_chat_completion_openai_v1,
        mode=instructor.Mode.JSON_SCHEMA,  # (2)!
    )

    return create


def generate_questions(create, chunks):
    questions_semantic_route = []

    for data in chunks:

        extraction_results = create(
            response_model=instructor.Partial[GenerateQuestion],  # (3)!
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a helpful assistant that generates questions related to documents and that can be answered in isolation.
    Generate 5 questions related to: \n{data}""",
                },
            ],
            # stream=True,
        )

        questions_semantic_route.extend([
            extraction_results.question1, 
            extraction_results.question2, 
            extraction_results.question3,
            extraction_results.question4, 
            extraction_results.question5,
            extraction_results.question6,
        ])

        print(questions_semantic_route)
        
    return questions_semantic_route

def main(file_path):
    create = initialize()
    text = read_pdf(file_path)
    chunks = chunk_text(text)
    questions = generate_questions(create, chunks)
    print(questions)


if __name__ == "__main__":
    
    main(file_path='/Users/thanhnguyen/Documents/Developer/multiple_routes/Llama3_Herd_of_Models.pdf')