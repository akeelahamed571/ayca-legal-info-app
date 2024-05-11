import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from io import BytesIO

load_dotenv()
# Paths to the PDF documents
DOCUMENT_PATHS = [
    r"D:\test_openai_RAG\CONVENTIONS-AGAINST-ILLICIT-TRAFFIC-IN-NARCOTIC-DRUGS-AND-PSYCHOTROPIC-SUBSTANCES-ACT-NO-1-OF-200.pdf",
    r"D:\test_openai_RAG\1-company-law-part-1-3-11-2019-notes.pdf",
    r"D:\test_openai_RAG\2-company-law-part-2-24-11-2019-notes.pdf",
    r"D:\test_openai_RAG\3-company-law-part-3-1-12-2019-notes.pdf",
    r"D:\test_openai_RAG\corporate-law.pdf"
    # Add paths for other PDF documents here
]

ALLOWED_EXTENSIONS = {'pdf'}

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_NAME = "gpt-4"
#llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
# Load OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_documents():
    global documents, embeddings
    documents = []
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    for path in DOCUMENT_PATHS:
        document = PdfReader(path)
        texts = []
        for page in document.pages:
            texts.append(page.extract_text())
        documents.append(texts)

    return documents

def answer_query(query):
    print('answer_query function')
    print("prompt", query)
    docs = load_documents()

    retrieved_docs = []
    for doc in docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents(doc)
        vectorstore = FAISS.from_documents(texts, embeddings)
        retrieved_docs.extend(vectorstore.similarity_search(query))

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print('\n')

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

    messages = [
        ("system", "You are a helpful assistant that Answer the legal question based on the provided context."),
        ("human", f"context: {context}\nQuestion: {query}"),
    ]
    
    print(messages)
    response = llm.invoke(messages)

    return response.content.strip()


@app.route("/legalQueries", methods=["GET"])
def index():
    return render_template("legalQuaries.html")

@app.route("/get_answer", methods=["POST"])
def answer_text_query():
    query = request.form["question"]
    answer = answer_query(query)
    return render_template("legalQuaries.html", answer=answer)

############################################################################################
def load_document_from_pdf(file):
    global document, embeddings
    text = []
    chunk_size = 1000
    chunk_overlap = 100

    document = PdfReader(file)
    for page in document.pages:
        text.append(page.extract_text())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents(text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore



def answer_legal_query(query, uploaded_file):
    print('answer_legal_query function')
    print("prompt", query)
    db = load_document_from_pdf(uploaded_file)

    retrieved_docs = db.similarity_search(query)  # Retrieve top relevant

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

    messages = [
        ("system", "You are a helpful assistant that answers legal questions based on the provided context."),
        ("human", f"context: {context}\nQuestion: {query}"),
    ]
    
    print(messages)
    response = llm.invoke(messages)

    return response.content.strip()
    
"""@app.route("/fileqna", methods=["GET", "POST"])
def file_qna():
    if request.method == "POST":
        if 'pdf_file' not in request.files:
            return "No file part"

        file = request.files['pdf_file']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            uploaded_file = BytesIO(file.read())
            query = request.form["question"]
            answer = answer_legal_query(query, uploaded_file)
            return render_template("fileqna.html", answer=answer)

    return render_template("fileqna.html") """  
    
import os
from werkzeug.utils import secure_filename

# Ensure the upload folder exists; create it if it doesn't
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/fileqna", methods=["GET", "POST"])
def file_qna():
    if request.method == "POST":
        if 'pdf_file' not in request.files:
            return "No file part"

        file = request.files['pdf_file']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            # Save the uploaded file to the upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)

            query = request.form["question"]
            answer = answer_legal_query(query, file_path)
            print("\n answer:",answer)
            return render_template("fileqna.html", answer=answer)

    return render_template("fileqna.html")



#################################################
#home page
@app.route("/home")
def home_page():
    return render_template("index.html")

# Route to render the footer.html template
@app.route('/footer')
def footer():
    return render_template('footer.html')

# Route to render the navbar.html template
@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

#################################################################
@app.route("/legalResources", methods=["GET", "POST"])
def leagalResources_page():
    return render_template("leagalResources.html")

##################################################################
#summarization page get
"""@app.route("/summarization")
def summarization():
    return render_template("summarization.html")"""


"""
#summarization page post
import spacy
#import pdfplumber
#from flask import Flask, render_template, request
#import os
#import torch
#from transformers import BartTokenizer, BartForConditionalGeneration
import openai


# Load BART model and tokenizer
#model_dir = "model_summarization"
#bart_tokenizer = BartTokenizer.from_pretrained(model_dir)
#bart_model = BartForConditionalGeneration.from_pretrained(model_dir)
openai.api_key = os.environ["OPENAI_API_KEY"] #"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"
# Load spaCy NER model

nlp_ner = spacy.load("model-best")  # load NER model

@app.route('/summarization', methods=['GET', 'POST'])
def summarization():
    if request.method == 'POST':
        file = request.files['pdf_file']

        # Read the contents of the uploaded PDF file
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
        except:
            return "Error parsing the PDF file"

        # Extract entities using spaCy NER model
        doc = nlp_ner(text)

        # Find the entity labeled as "JUDGMENT_1"
        extracted_entity = None
        for ent in doc.ents:
            if ent.label_ == "JUDGEMENT_1":
                extracted_entity = ent.text
                print(extracted_entity)
                break

        # If "JUDGEMENT_1" entity is found, use it as the context for summarization
        if extracted_entity:
            # Send the extracted judgment to GPT-3.5 Turbo for summarization
            legal_document_text = extracted_entity[:4097]
            # Initialize the conversation with the prompt
            messages = [
                {"role": "user", "content": "Please summarize ( elaborately) for this large legal text from the judgment of the main judge. Please provide me with a comprehensive summary that covers every major point, including the final decision by the judge and the court,location(mention court names),who are against whom(mention the names).Ensure that the summary is explaining every details of the case and easy to understand for lawyers, highlighting the key aspects of the judgment."},
                {"role": "assistant", "content": legal_document_text}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.4,#0.7
                max_tokens=800,#256
                messages=messages,
                stop=["\n"]
                #input=legal_document_text
            )
            print("################################################")
            print("reached create summary")
            # Extract the summarized text from the GPT-3.5 Turbo response
            # Retrieve the generated text from the response
            generated_text = response.choices[0].message['content'] #text
            print("################################################")
            print("summmary text:",generated_text)

            # Return the summarization template with entity and summary
            return render_template('summarization.html', entity=extracted_entity, summary=generated_text)
        else:
            return "Judgment entity not found in the uploaded PDF file"

    return render_template('summarization.html')"""
#from langchain_openai import ChatOpenAI
#from flask import Flask, render_template, request
#import os
"""import spacy
import pdfplumber"""

# Initialize ChatOpenAI
#chat_openai = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load spaCy NER model
#lp_ner = spacy.load("model-best")  # load NER model

##############################################################AFTEEr
""""@app.route("/summarization", methods=['GET', 'POST'])
def summarization():
    if request.method == 'POST':
        file = request.files['pdf_file']
        model="gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model, temperature=0,api_key=os.environ["OPENAI_API_KEY"])
        # Read the contents of the uploaded PDF file
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
        except:
            return "Error parsing the PDF file"

        # Extract entities using spaCy NER model
        doc = nlp_ner(text)

        # Find the entity labeled as "JUDGMENT_1"
        extracted_entity = None
        for ent in doc.ents:
            if ent.label_ == "JUDGEMENT_1":
                extracted_entity = ent.text
                print(extracted_entity)
                break

        # If "JUDGMENT_1" entity is found, use it as the context for summarization
        if extracted_entity:
            # Send the extracted judgment to GPT-3.5 Turbo for summarization
            legal_document_text = extracted_entity[:4097]
            # Initialize the conversation with the prompt
            messages = [
                {"role": "user", "content": "Please summarize ( elaborately) for this large legal text from the judgment of the main judge. Please provide me with a comprehensive summary that covers every major point, including the final decision by the judge and the court,location(mention court names),who are against whom(mention the names).Ensure that the summary is explaining every details of the case and easy to understand for lawyers, highlighting the key aspects of the judgment."},
                {"role": "assistant", "content": legal_document_text}
            ]
            # Generate the summary using ChatOpenAI
            response =  llm.invoke(messages)

            # Extract the summarized text from the response
            generated_text =response.content.strip()# response['choices'][0]['message']['content']
            print("################################################")
            print("summary text:", generated_text)

            # Return the summarization template with entity and summary
            return render_template('summarization.html', entity=extracted_entity, summary=generated_text)
        else:
            return "Judgment entity not found in the uploaded PDF file"

    return render_template('summarization.html')

"""






#########################################################################################################






if __name__ == "__main__":
    app.run(debug=False)
