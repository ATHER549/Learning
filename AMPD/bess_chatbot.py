import os
import base64
from typing import List
from dotenv import load_dotenv
import fitz
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def extract_pdf_content(pdf_file):
    """Extract text and images from PDF file"""
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"Cannot find PDF file: {pdf_file}")
    
    doc = fitz.open(pdf_file)
    all_content = []
    
    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    print(f"Processing {len(doc)} pages...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"Page {page_num + 1}...")
        
        # Get text content
        page_text = page.get_text()
        if page_text.strip():
            chunks = text_splitter.split_text(page_text)
            for i, chunk in enumerate(chunks):
                all_content.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": "text",
                        "page": page_num + 1,
                        "chunk": i
                    }
                ))
        
        # Process images
        images = page.get_images(full=True)
        for img_num, img in enumerate(images):
            try:
                image_data = doc.extract_image(img[0])
                img_bytes = image_data["image"]
                
                # Skip tiny images (probably icons)
                if len(img_bytes) < 10000:
                    continue
                    
                print(f"  Analyzing image {img_num + 1}...")
                description = analyze_image(img_bytes)
                
                all_content.append(Document(
                    page_content=description,
                    metadata={
                        "source": "image",
                        "page": page_num + 1,
                        "image": img_num
                    }
                ))
            except Exception as e:
                print(f"  Skipped image {img_num + 1}: {e}")
                continue
    
    doc.close()
    print(f"Extracted {len(all_content)} content pieces")
    return all_content

def analyze_image(image_bytes):
    """Get description of image using Azure OpenAI vision"""
    # Configure Azure OpenAI with proper timeout and retry settings
    vision_model = AzureChatOpenAI(
        deployment_name="gpt-4o",
        max_tokens=300,
        temperature=0,
        timeout=60,
        max_retries=2
    )
    
    # Convert image to base64
    img_b64 = base64.b64encode(image_bytes).decode()
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "Describe this technical diagram from a battery system manual. Focus on components, connections, and safety warnings. Keep it brief."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            }
        ]
    }]
    
    try:
        response = vision_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"Image from manual (analysis failed: {str(e)[:50]})"

def setup_vector_database(documents, db_folder="./vector_db"):
    """Create or load vector database for document search"""
    
    # Azure OpenAI embeddings with proper configuration
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_deployment="text-embedding-3-small",
        chunk_size=1000  # Azure best practice for batch processing
    )
    
    if os.path.exists(db_folder):
        print(f"Loading existing database from {db_folder}")
        vectordb = Chroma(
            persist_directory=db_folder, 
            embedding_function=embeddings
        )
    else:
        print(f"Creating new database in {db_folder}")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_folder
        )
    
    return vectordb.as_retriever(search_kwargs={"k": 4})

def build_qa_system(retriever):
    """Build the question-answering system"""
    
    # Simple, direct prompt
    prompt_template = """Answer the question using only the manual content below. 
If you can't find the answer, just say so.

Manual content:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Configure Azure OpenAI for responses
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o", 
        temperature=0,
        max_tokens=500
    )
    
    # Build the chain
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

def run_chatbot():
    """Main chatbot function"""
    
    # Check Azure API key
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: Missing AZURE_OPENAI_API_KEY in environment")
        print("Add it to your .env file")
        return
    
    manual_file = "AmpD Enertainer User Manual (NCM) - Rev 2.3.pdf"
    database_path = "./bess_vector_db"
    
    # Process PDF if database doesn't exist
    if not os.path.exists(database_path):
        try:
            print("First time setup - processing manual...")
            content = extract_pdf_content(manual_file)
            setup_vector_database(content, database_path)
            print("Setup complete!")
        except FileNotFoundError:
            print(f"Can't find {manual_file}")
            print("Make sure the PDF is in the same folder as this script")
            return
        except Exception as e:
            print(f"Setup failed: {e}")
            return
    
    # Load the QA system
    retriever = setup_vector_database([], database_path)
    qa_system = build_qa_system(retriever)
    
    print("\nBESS Manual Assistant Ready!")
    print("Ask questions about the AmpD Enertainer")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            question = input("Your question: ")
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            if not question.strip():
                continue
                
            print("Searching manual...")
            answer = qa_system.invoke(question)
            print(f"\nAnswer: {answer}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Something went wrong: {e}")
            continue

if __name__ == "__main__":
    run_chatbot()
