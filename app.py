import base64
import io
from PyPDF2 import PdfReader, PdfWriter
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import mysql.connector
import openai
from openai import AzureOpenAI
import logging
import sys
import boto3
import traceback
import re
import json
from io import BytesIO
from llama_index.core import VectorStoreIndex, Document, ServiceContext, PromptTemplate, StorageContext, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings
from PyPDF2 import PdfReader
import tempfile
import chromadb
from chromadb.errors import InvalidCollectionException
from llama_index.vector_stores.chroma import ChromaVectorStore
import hashlib
from dotenv import load_dotenv

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Access AWS credentials
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
s3_bucket_name = os.getenv('S3_BUCKET_NAME')
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

# MySQL database credentials
db_config_cv_parser = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# GPT-3.5 credentials
api_key1 = os.getenv('GPT35_API_KEY')
azure_endpoint1 = os.getenv('GPT35_ENDPOINT')
api_version1 = os.getenv('GPT35_API_VERSION')

# GPT-4o mini model credentials
api_key = os.getenv('GPT4O_API_KEY')
azure_endpoint = os.getenv('GPT4O_ENDPOINT')
api_version = os.getenv('GPT4O_API_VERSION')

#Initialize LLM model
llm = AzureOpenAI(
    model="gpt-4o-mini",
    deployment_name="omni_model",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

#Initialize embedding model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="openai_embedding-ada-002",
    api_key=api_key1,
    azure_endpoint=azure_endpoint1,
    api_version=api_version1,
)
Settings.llm = llm
Settings.embed_model = embed_model

# Function to establish MySQL connection for the second database (cv_parser)
def get_db_connection_cv_parser():
    conn = None
    try:
        conn = mysql.connector.connect(**db_config_cv_parser)
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL database (cv_parser): {e}")
    return conn

def get_collection_name(role_id, cv_id):
    return f"role_{role_id}_cv_{cv_id}"

def get_or_create_collection(chroma_client, collection_name, role_id):
    """Helper function to get or create a collection"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except InvalidCollectionException:
        # Collection doesn't exist, create it
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"role_id": str(role_id)}
        )
    return collection

def backup_collection_to_s3(collection, s3_client, role_id, cv_id):
    """
    Backup a ChromaDB collection to S3 in centelon-generative-ai bucket
    under CV_parser_testing_embedding/Embedding_Collections/
    """
    try:
        # Get all data from the collection
        collection_data = collection.get()
        
        # Create a structured format for the embedding data
        embedding_data = {
            'ids': collection_data['ids'],
            'embeddings': collection_data['embeddings'],
            'documents': collection_data['documents'],
            'metadatas': collection_data['metadatas']
        }
        
        # Convert to JSON
        json_data = json.dumps(embedding_data)
        
        # Define the S3 path with the new folder structure
        s3_path = f"CV_parser_testing_embedding/Embedding_Collections/role_{role_id}/cv_{cv_id}/embeddings.json"
        
        # Upload to S3
        s3_client.put_object(
            Bucket='centelon-generative-ai',
            Key=s3_path,
            Body=json_data
        )
        
        print(f"Successfully backed up embeddings for CV {cv_id} to S3 at path: {s3_path}")
        return True
        
    except Exception as e:
        print(f"Error backing up embeddings to S3: {str(e)}")
        traceback.print_exc()
        return False

def load_embeddings_from_s3(s3_client, role_id, cv_id):
    """
    Load embeddings from S3 from the specified path in centelon-generative-ai bucket
    """
    try:
        # Define the S3 path with the new folder structure
        s3_path = f"CV_parser_testing_embedding/Embedding_Collections/role_{role_id}/cv_{cv_id}/embeddings.json"
        
        # Get the object from S3
        response = s3_client.get_object(
            Bucket='centelon-generative-ai',
            Key=s3_path
        )
        
        # Read and parse the JSON data
        json_data = response['Body'].read().decode('utf-8')
        embedding_data = json.loads(json_data)
        
        return embedding_data
        
    except Exception as e:
        print(f"Error loading embeddings from S3: {str(e)}")
        traceback.print_exc()
        return None
    
def clear_existing_embeddings(chroma_client, role_id, cv_id):
    try:
        collection_name = get_collection_name(role_id)
        collection = chroma_client.get_collection(name=collection_name)
        # Delete embeddings for specific cv_id
        collection.delete(
            where={"cv_id": str(cv_id)}
        )
        print(f"Cleared existing embeddings for CV {cv_id} in collection {collection_name}")
    except Exception as e:
        print(f"Error clearing embeddings: {str(e)}")

def clean_base64(s):
    return re.sub(r'\s+', '', s)

def pad_base64(s):
    return s + '=' * (-len(s) % 4)
	
@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the API'})

# Initialize database
def initialize_databases():
    # Initialize flask_batch_mode database
    conn_cv_parser = get_db_connection_cv_parser()
    if conn_cv_parser:
        cursor = conn_cv_parser.cursor()
        try:
            # Create tables for flask_batch_mode
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chapters (
                    ChapterID INT PRIMARY KEY,
                    ChapterName VARCHAR(255) UNIQUE NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    role_id INT PRIMARY KEY,
                    role_name VARCHAR(255) NOT NULL,
                    ChapterID INT NOT NULL,
                    CONSTRAINT fk_Chapter FOREIGN KEY (ChapterID) REFERENCES chapters(ChapterID),
                    CONSTRAINT uc_RoleChapter UNIQUE (role_name, ChapterID)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applicants (
                    cv_id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    chapter_name VARCHAR(255) NOT NULL,
                    position_name VARCHAR(255) NOT NULL,
                    role_id INT,
                    file_path VARCHAR(255) NULL,
                    status VARCHAR(50) DEFAULT 'In-Process',
                    FOREIGN KEY (role_id) REFERENCES roles(role_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS all_questions (
                    QuestionID INT AUTO_INCREMENT PRIMARY KEY,
                    role_id INT,
                    ChapterName VARCHAR(255) NOT NULL,
                    questions TEXT,
                    FOREIGN KEY (role_id) REFERENCES roles(role_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    cv_id INT,
                    name VARCHAR(255),
                    role_id INT,
                    question TEXT,
                    answer TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS result_table (
                    cv_id INT,
                    name VARCHAR(255),
                    role_id INT,
                    conclusion VARCHAR(20) NOT NULL,
                    UNIQUE KEY cv_id_unique (cv_id)
                )
            """)
            
            conn_cv_parser.commit()
            print("flask_batch_mode database initialized successfully")
        except mysql.connector.Error as e:
            print(f"Error initializing flask_batch_mode database: {e}")
        finally:
            cursor.close()
            conn_cv_parser.close()

def process_pdf_content(pdf_bytes, cv_id):
    """Validate and process PDF content"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(pdf_reader.pages) == 0:
            raise ValueError("PDF has no pages")
        return True
    except Exception as e:
        app.logger.error(f"Error validating PDF for CV {cv_id}: {str(e)}")
        return False

# Call this function to initialize database
initialize_databases()

@app.route('/candidate', methods=['GET', 'POST'])
def candidate():
    if request.method == 'POST':
        try:
            data = request.json if request.is_json else request.form
            app.logger.info(f"Received POST data: {data}")
            
            name = data.get('name', '').strip().replace(" ", "_")
            chapter_select = data.get('chapter_select', '')
            position_select = data.get('position_select', '')
            
            if not name or not chapter_select or not position_select:
                return jsonify({"error": "Missing required fields"}), 400

            # Handle both file upload and base64 string
            pdf_bytes = None
            if request.files and 'file' in request.files:
                # Direct PDF file upload
                file = request.files['file']
                if file and file.filename.lower().endswith('.pdf'):
                    pdf_bytes = file.read()
                else:
                    return jsonify({"error": "Invalid file format. Only PDF files are accepted."}), 400
            else:
                # Base64 encoded PDF
                file_content = data.get('file', '')
                if file_content:
                    app.logger.info(f"Received base64 string of length: {len(file_content)}")
                    try:
                        # Clean and pad the base64 string
                        file_content = clean_base64(file_content)
                        file_content = pad_base64(file_content)
                        pdf_bytes = base64.b64decode(file_content)
                        
                        # Validate PDF format
                        try:
                            PdfReader(io.BytesIO(pdf_bytes))
                        except Exception as e:
                            app.logger.error(f"Invalid PDF format: {str(e)}")
                            return jsonify({"error": "Invalid PDF format in base64 string"}), 400
                            
                    except Exception as e:
                        app.logger.error(f"Error decoding base64: {str(e)}")
                        return jsonify({"error": "Invalid base64 encoded PDF"}), 400

            if not pdf_bytes:
                return jsonify({"error": "No PDF file provided"}), 400

            app.logger.info(f"Processing request for {name}, chapter: {chapter_select}, position: {position_select}")

            try:
                # Create a PDF file from the bytes content
                pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                pdf_writer = PdfWriter()

                # Validate PDF content
                if len(pdf_reader.pages) == 0:
                    return jsonify({"error": "PDF file is empty"}), 400

                for page in range(len(pdf_reader.pages)):
                    pdf_writer.add_page(pdf_reader.pages[page])

                # Create a temporary file to store the PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    pdf_writer.write(temp_file)
                    temp_file_path = temp_file.name

                # Save to database first to get cv_id
                conn = get_db_connection_cv_parser()
                if not conn:
                    return jsonify({"error": "Database connection failed"}), 500

                cursor = conn.cursor()
                try:
                    # Get role_id for the position and chapter
                    query = """
                        INSERT INTO applicants (name, chapter_name, position_name, role_id, file_path, status)
                        SELECT %s, %s, %s, role_id, NULL, 'In-Process'
                        FROM roles
                        WHERE role_name = %s AND chapter_id = (
                            SELECT chapter_id FROM chapters WHERE chapter_name = %s
                        )
                    """
                    cursor.execute(query, (name, chapter_select, position_select, position_select, chapter_select))
                    conn.commit()
                    cv_id = cursor.lastrowid

                    if not cv_id:
                        return jsonify({"error": "Failed to create applicant record"}), 500

                    # Now create file_name with cv_id
                    file_name = f"{cv_id}_{name}.pdf"
                    file_path = f"CV_Parser/{chapter_select}/{position_select}/{file_name}"

                    # Upload to S3
                    try:
                        s3_client.upload_file(temp_file_path, s3_bucket_name, file_path)
                        app.logger.info(f"PDF uploaded to S3 bucket successfully: {file_path}")

                        # Update the file_path in the database
                        update_query = "UPDATE applicants SET file_path = %s WHERE cv_id = %s"
                        cursor.execute(update_query, (file_path, cv_id))
                        conn.commit()

                        app.logger.info(f"Application for {name} submitted successfully with cv_id: {cv_id}")
                        
                        return jsonify({
                            'message': 'Application submitted successfully',
                            'cv_id': cv_id,
                            'name': name,
                            'chapter': chapter_select,
                            'position': position_select
                        })

                    except Exception as e:
                        app.logger.error(f"Error uploading file to S3 bucket: {str(e)}")
                        # Delete the applicant record if S3 upload fails
                        cursor.execute("DELETE FROM applicants WHERE cv_id = %s", (cv_id,))
                        conn.commit()
                        return jsonify({'error': 'Failed to upload file to S3'}), 500

                except mysql.connector.Error as e:
                    app.logger.error(f"Database error: {str(e)}")
                    return jsonify({'error': 'Database error occurred'}), 500
                finally:
                    cursor.close()
                    conn.close()
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        app.logger.error(f"Error deleting temporary file: {str(e)}")

            except Exception as e:
                app.logger.error(f"Error processing PDF: {str(e)}")
                return jsonify({"error": f"Invalid PDF file: {str(e)}"}), 400

        except Exception as e:
            app.logger.error(f"Unexpected error in candidate route: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': 'An unexpected error occurred'}), 500

    elif request.method == 'GET':
        # Handling GET request to fetch chapters and positions
        conn = get_db_connection_cv_parser()
        if conn:
            cursor = conn.cursor()
            try:
                # Fetch chapters
                cursor.execute("SELECT ChapterName FROM chapters")
                chapters = cursor.fetchall()
                chapter_names = [row[0] for row in chapters]
                
                # Fetch positions/roles
                cursor.execute("SELECT role_name FROM roles")
                roles = cursor.fetchall()
                position_names = [row[0] for row in roles]

                return jsonify({
                    'chapter_names': chapter_names, 
                    'position_names': position_names
                })
            except mysql.connector.Error as e:
                app.logger.error(f"Error fetching chapters and positions: {e}")
                return jsonify({'error': 'An error occurred while fetching data'}), 500
            finally:
                cursor.close()
                conn.close()
        else:
            return jsonify({'error': 'Failed to establish database connection'}), 500


@app.route('/organization', methods=['GET', 'POST'])
def organization():
    if request.method == 'POST':
        action = request.json.get('action')
        if action == 'save':
            chapter_select = request.json['chapter_select']
            position_select = request.json['position_select']
            new_question = request.json['new_question']
            
            conn = get_db_connection_cv_parser()
            if not conn:
                return jsonify({'error': 'Database connection failed'}), 500
                
            cursor = conn.cursor()
            try:
                # First get the role_id for the given chapter and position
                cursor.execute("""
                    SELECT r.role_id
                    FROM roles r
                    INNER JOIN chapters c ON r.chapter_id = c.chapter_id
                    WHERE c.chapter_name = %s AND r.role_name = %s
                """, (chapter_select, position_select))
                
                role_result = cursor.fetchone()
                if not role_result:
                    cursor.close()
                    conn.close()
                    return jsonify({'error': 'Role not found'}), 404
                
                role_id = role_result[0]
                
                # Insert the new question
                cursor.execute("""
                    INSERT INTO all_questions (role_id, ChapterName, Questions)
                    VALUES (%s, %s, %s)
                """, (role_id, chapter_select, new_question))
                
                conn.commit()
                question_id = cursor.lastrowid
                
                return jsonify({
                    'message': 'New question added successfully',
                    'question_id': question_id
                })
                
            except mysql.connector.Error as e:
                print(f"Error adding new question to database: {e}")
                return jsonify({'error': f'Database error: {str(e)}'}), 500
            finally:
                cursor.close()
                conn.close()
                
        elif 'chapter' in request.json:
            selected_chapter = request.json['chapter']
            positions = []
            
            conn = get_db_connection_cv_parser()
            if not conn:
                return jsonify({'error': 'Database connection failed'}), 500
                
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT role_name FROM roles r
                    INNER JOIN chapters c ON r.chapter_id = c.chapter_id
                    WHERE c.chapter_name = %s
                """, (selected_chapter,))
                positions = [row[0] for row in cursor.fetchall()]
                return jsonify({'positions': positions})
                
            except mysql.connector.Error as e:
                print(f"Error fetching positions: {e}")
                return jsonify({'error': f'Database error: {str(e)}'}), 500
            finally:
                cursor.close()
                conn.close()
        
        return jsonify({'error': 'Invalid action'}), 400

    # Handle GET request
    chapter_select = request.args.get('chapter_select')
    position_select = request.args.get('position_select')
    
    conn = get_db_connection_cv_parser()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
        
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT chapter_name FROM chapters")
        chapters = cursor.fetchall()
        chapter_names = [row[0] for row in chapters]

        cursor.execute("SELECT role_name FROM roles")
        roles = cursor.fetchall()
        position_names = [row[0] for row in roles]

        existing_questions = []
        if chapter_select and position_select:
            cursor.execute("""
                SELECT q.QuestionID, q.Questions
                FROM all_questions q
                INNER JOIN roles r ON q.role_id = r.role_id
                INNER JOIN chapters c ON r.chapter_id = c.chapter_id
                WHERE c.chapter_name = %s AND r.role_name = %s
            """, (chapter_select, position_select))
            existing_questions = cursor.fetchall()

        return jsonify({
            'chapter_names': chapter_names,
            'position_names': position_names,
            'existing_questions': existing_questions
        })
        
    except mysql.connector.Error as e:
        print(f"Error fetching data: {e}")
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/delete_question', methods=['POST'])
def delete_question():
    question_id = request.json.get('question_id')
    chapter_select = request.json.get('chapter_select')
    position_select = request.json.get('position_select')
    if question_id:
        conn = get_db_connection_cv_parser()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM all_questions WHERE QuestionID = %s", (question_id,))
                conn.commit()
                print(f"Question with ID {question_id} deleted successfully")
            except mysql.connector.Error as e:
                print(f"Error deleting question with ID {question_id}: {e}")
            finally:
                cursor.close()
                conn.close()
    return jsonify({'message': 'Question deleted successfully'})

@app.route('/run_parser', methods=['POST'])
def run_parser():
    conn = get_db_connection_cv_parser()
    if not conn:
        return jsonify({'error': 'Failed to establish database connection'}), 500

    cursor = conn.cursor()
    try:
        # Fetch all applicants with 'In-Process' status
        cursor.execute("""
            SELECT cv_id, name, role_id, file_path
            FROM applicants
            WHERE status = 'In-Process'
        """)
        applicants = cursor.fetchall()
        
        if not applicants:
            return jsonify({'message': 'No CVs to process'}), 200

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Set up Settings for llama-index
        Settings.llm = llm
        Settings.embed_model = embed_model

        for cv_id, name, role_id, file_path in applicants:
            print(f"\nProcessing CV {cv_id} for {name}...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Clear existing embeddings for this CV
                    collection_name = f"role_{role_id}_cv_{cv_id}"
                    try:
                        existing_collection = chroma_client.get_collection(name=collection_name)
                        chroma_client.delete_collection(collection_name)
                        print(f"Deleted existing collection: {collection_name}")
                    except Exception as e:
                        print(f"No existing collection to delete: {str(e)}")

                    # Create fresh collection
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        metadata={"role_id": str(role_id), "cv_id": str(cv_id)}
                    )
                    print(f"Created new collection: {collection_name}")

                    # Download the CV from S3
                    print(f"Downloading CV from S3...")
                    local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                    s3_client.download_file(s3_bucket_name, file_path, local_file_path)
                    
                    # Load document text
                    print("Loading document text...")
                    documents = SimpleDirectoryReader(temp_dir).load_data()
                    
                    # Create a unique document ID
                    doc_id = f"cv_{cv_id}"
                    
                    # Create embeddings and store in ChromaDB
                    print("Creating and storing embeddings...")
                    for idx, doc in enumerate(documents):
                        doc_text = doc.text
                        # Generate embedding using the embed_model
                        embedding = embed_model.get_text_embedding(doc_text)
                        
                        # Store in ChromaDB with unique ID and metadata
                        collection.add(
                            ids=[f"{doc_id}_chunk_{idx}"],
                            embeddings=[embedding],
                            documents=[doc_text],
                            metadatas=[{
                                "cv_id": str(cv_id),
                                "name": name,
                                "chunk_id": str(idx),
                                "role_id": str(role_id)
                            }]
                        )
                    
                    # Backup embeddings to S3
                    print("Backing up embeddings to S3...")
                    backup_success = backup_collection_to_s3(collection, s3_client, role_id, cv_id)
                    if backup_success:
                        print(f"Successfully backed up embeddings to S3")
                    else:
                        print(f"Failed to backup embeddings to S3")

                    # Create vector store and index
                    print("Creating vector store and index...")
                    vector_store = ChromaVectorStore(chroma_collection=collection)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    # Initialize the base vector index
                    vector_index = VectorStoreIndex(
                        documents,
                        storage_context=storage_context
                    )
                    
                    # Create vector query engine with specific prompt template
                    query_template = (
                        "Based on the provided CV content, please answer the following question "
                        "about the candidate. If the information is clearly present in the CV, "
                        "answer with just 'Yes' or 'No'. If the answer requires explanation or "
                        "the information is not directly stated, provide a detailed response: \n"
                        "Question: {query_str}\n"
                    )
                    
                    vector_query_engine = vector_index.as_query_engine(
                        text_qa_template=PromptTemplate(query_template),
                        response_mode="tree_summarize",
                        use_async=True
                    )
                    
                    # Create query engine tool
                    vector_tool = QueryEngineTool.from_defaults(
                        query_engine=vector_query_engine,
                        description=f"CV analysis tool for {name}'s resume"
                    )
                    
                    # Create tool mapping and object index
                    tool_mapping = SimpleToolNodeMapping.from_objects([vector_tool])
                    obj_index = ObjectIndex.from_objects(
                        [vector_tool],
                        tool_mapping,
                        VectorStoreIndex
                    )
                    
                    # Create the final query engine
                    query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

                    # Fetch questions for this role
                    print("Fetching questions...")
                    cursor.execute("""
                        SELECT questions
                        FROM all_questions
                        WHERE role_id = %s
                    """, (role_id,))
                    questions = cursor.fetchall()
                    
                    if not questions:
                        print(f"No questions found for role_id {role_id}")
                        continue
                    
                    print(f"Processing {len(questions)} questions...")
                    answers = []
                    for i, question in enumerate(questions):
                        question_text = question[0].strip()
                        print(f"\nProcessing question {i+1}: {question_text}")
                        
                        try:
                            # Query using the tool retriever router query engine
                            response = query_engine.query(question_text)
                            answer = str(response)
                            answers.append(answer)
                            print(f"Answer received: {answer[:100]}...")  # Print first 100 chars
                            
                            # Store in chat history
                            cursor.execute("""
                                INSERT INTO chat_history (cv_id, name, role_id, question, answer)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (cv_id, name, role_id, question_text, answer))
                            conn.commit()
                            print("Answer stored in chat history")
                        except Exception as e:
                            print(f"Error processing question: {str(e)}")
                            answers.append("Error processing question")
                    
                    # Calculate status
                    print("Calculating final status...")
                    status = calculate_status(answers)
                    print(f"Final status: {status}")
                    
                    # Update result table
                    cursor.execute("""
                        INSERT INTO result_table (cv_id, name, role_id, conclusion)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        conclusion = VALUES(conclusion)
                    """, (cv_id, name, role_id, status))
                    
                    # Update applicant status
                    cursor.execute("""
                        UPDATE applicants
                        SET status = 'Processed'
                        WHERE cv_id = %s
                    """, (cv_id,))
                    
                    conn.commit()
                    print(f"Successfully completed processing CV {cv_id} for {name}")
                    
                except Exception as e:
                    print(f"Error processing CV {cv_id}: {str(e)}")
                    traceback.print_exc()
                    cursor.execute("""
                        UPDATE applicants
                        SET status = 'Error'
                        WHERE cv_id = %s
                    """, (cv_id,))
                    conn.commit()
                finally:
                    # Clean up by deleting temporary files
                    if 'local_file_path' in locals() and os.path.exists(local_file_path):
                        try:
                            os.remove(local_file_path)
                        except Exception as e:
                            print(f"Error removing temporary file: {str(e)}")

        return jsonify({'message': f'Processed {len(applicants)} CVs successfully'})

    except Exception as e:
        print(f"Error in run_parser: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during batch processing'}), 500
    finally:
        cursor.close()
        conn.close()

def calculate_status(answers):
   # Print all answers and count only Yes/No
   yes_count = 0
   no_count = 0
   
   print("\nAnalyzing Answers:")
   for i, answer in enumerate(answers, 1):
       print(f"\nAnswer {i}: {answer}")
       
       # Check if answer is just "Yes" or "No"
       if answer.lower().strip() == "yes":
           yes_count += 1
           print("Counted as: Yes")
       elif answer.lower().strip() == "no":
           no_count += 1
           print("Counted as: No") 
       else:
           print("Not counted (descriptive answer)")
   
   total_responses = yes_count + no_count
   
   print(f"\nFinal Results:")
   print(f"Yes count: {yes_count}")
   print(f"No count: {no_count}")
   print(f"Total Yes/No responses: {total_responses}")
   
   if total_responses == 0:
       print("Status: Inconclusive (no Yes/No answers found)")
       return "Inconclusive"
       
   percentage_yes = (yes_count / total_responses) * 100
   print(f"Percentage yes: {percentage_yes}%")
   
   if no_count == 0:
       print("Status: Selected (all answers were Yes)")
       return "Selected"
   elif yes_count == 0:
       print("Status: Rejected (all answers were No)")
       return "Rejected"
   elif percentage_yes >= 60:
       print("Status: Selected (>= 60% Yes answers)")
       return "Selected"
   else:
       print("Status: Rejected (< 60% Yes answers)")
       return "Rejected"

@app.route('/view', methods=['GET'])
def view():
    conn = get_db_connection_cv_parser()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT cv_id, name, chapter_name, position_name, role_id, status, file_path FROM applicants")
            applicants = cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error fetching applicants: {e}")
            applicants = []
        finally:
            cursor.close()
            conn.close()
    return jsonify(applicants)

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        cv_id = request.json.get('cv_id')
        if not cv_id:
            return jsonify({'error': 'cv_id is required'}), 400

        conn = get_db_connection_cv_parser()
        if conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute("SELECT * FROM result_table WHERE cv_id = %s", (cv_id,))
                result = cursor.fetchone()
                if result:
                    return jsonify(result)
                else:
                    return jsonify({'message': 'No results found for the given cv_id'}), 404
            except mysql.connector.Error as e:
                app.logger.error(f"Error fetching result for cv_id {cv_id}: {e}")
                return jsonify({'error': 'An error occurred while fetching results'}), 500
            finally:
                cursor.close()
                conn.close()
    else:  # GET request
        conn = get_db_connection_cv_parser()
        if conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute("SELECT * FROM result_table")
                result_table = cursor.fetchall()
                return jsonify(result_table)
            except mysql.connector.Error as e:
                app.logger.error(f"Error fetching result table: {e}")
                return jsonify({'error': 'An error occurred while fetching results'}), 500
            finally:
                cursor.close()
                conn.close()

    return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    cv_id = request.json.get('cv_id')
    if cv_id.isdigit():
        cv_id = int(cv_id)
        conn = get_db_connection_cv_parser()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM chat_history WHERE cv_id = %s", (cv_id,))
            chat_history = cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error fetching chat history: {e}")
            chat_history = []
        finally:
            cursor.close()
            conn.close()
        return jsonify(chat_history)
    else:
        return jsonify({'error': 'Please enter a valid numerical CV ID.'})

@app.route('/chapters', methods=['POST'])
def handle_chapters():
    chapter_id = request.json.get('chapter_id')
    chapter_name = request.json.get('chapter_name')
    
    conn = get_db_connection_cv_parser()
    if conn:
        cursor = conn.cursor()
        try:
            # Check if the chapter already exists
            cursor.execute("SELECT * FROM chapters WHERE ChapterID = %s OR ChapterName = %s", (chapter_id, chapter_name))
            existing_chapter = cursor.fetchone()
            
            if existing_chapter:
                return jsonify({'message': 'Chapter already exists', 'chapter_id': existing_chapter[0]})
            else:
                # Insert the new chapter with the provided ID
                cursor.execute("INSERT INTO chapters (ChapterID, ChapterName) VALUES (%s, %s)", (chapter_id, chapter_name))
                conn.commit()
                return jsonify({'message': 'Chapter added successfully', 'chapter_id': chapter_id})
        except mysql.connector.Error as e:
            print(f"Error handling chapters: {e}")
            return jsonify({'error': 'An error occurred while handling chapters'}), 500
        finally:
            cursor.close()
            conn.close()
    else:
        return jsonify({'error': 'Failed to establish database connection'}), 500

@app.route('/roles', methods=['POST'])
def handle_roles():
    role_id = request.json.get('role_id')
    role_name = request.json.get('role_name')
    chapter_id = request.json.get('chapter_id')
    
    conn = get_db_connection_cv_parser()
    if conn:
        cursor = conn.cursor()
        try:
            # Check if the chapter exists
            cursor.execute("SELECT * FROM chapters WHERE ChapterID = %s", (chapter_id,))
            chapter = cursor.fetchone()
            if not chapter:
                return jsonify({'error': 'Chapter does not exist'}), 404

            # Check if the role already exists for this chapter
            cursor.execute("SELECT * FROM roles WHERE role_id = %s OR (role_name = %s AND ChapterID = %s)", (role_id, role_name, chapter_id))
            existing_role = cursor.fetchone()
            
            if existing_role:
                return jsonify({'message': 'Role already exists', 'role_id': existing_role[0]})
            else:
                # Insert the new role with the provided ID
                cursor.execute("INSERT INTO roles (role_id, role_name, ChapterID) VALUES (%s, %s, %s)", (role_id, role_name, chapter_id))
                conn.commit()
                return jsonify({'message': 'Role added successfully', 'role_id': role_id})
        except mysql.connector.Error as e:
            print(f"Error handling roles: {e}")
            return jsonify({'error': 'An error occurred while handling roles'}), 500
        finally:
            cursor.close()
            conn.close()
    else:
        return jsonify({'error': 'Failed to establish database connection'}), 500

if __name__ == '__main__':
    app.run()