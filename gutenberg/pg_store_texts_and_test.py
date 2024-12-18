from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import get_text_by_id
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from supabase import create_client, Client
from supabase.client import ClientOptions
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_HTTPS_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COOKING_KEYWORDS = ["cooking", "recipes", "cookbook", "culinary"]

# Initialize Supabase and OpenAI embeddings
supabase_client : Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(
    postgrest_client_timeout=360,
    storage_client_timeout=360,
    schema="public",
  ))

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Gutenberg cache
cache = GutenbergCache.get_cache()

def search_gutenberg_titles(keywords):
    """Search Project Gutenberg for books matching cooking-related keywords."""
    matching_books = []

    # Build the SQL WHERE clause for matching keywords in titles
    keyword_filters = " OR ".join([f"t.name LIKE '%{keyword}%'" for keyword in keywords])

    # Query the SQLite database for matching titles and their Gutenberg book IDs
    query = (f"""
        SELECT b.gutenbergbookid, t.name 
        FROM titles t
        JOIN books b ON t.bookid = b.id
        WHERE {keyword_filters}
    """)

    # Execute the query
    results = cache.native_query(query)

    # Process results into a list of tuples (gutenbergbookid, title)
    for row in results:
        gutenbergbookid, title = row
        matching_books.append((gutenbergbookid, title))

    return matching_books

def download_and_store_books(matching_books):
    """Download books, split text, generate embeddings, and store in Supabase."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = []

    for book_id, title in matching_books:
        print(f"Processing: {title} (ID: {book_id})")
        try:
            # Download book content
            raw_text = get_text_by_id(book_id)
            content = raw_text.decode("utf-8", errors="ignore")  # Decode to string

            # Split the text into manageable chunks
            chunks = text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                # Construct metadata as a JSON object
                metadata = {
                    "title": title,
                    "gutenberg_id": str(book_id),
                    "chunk_index": i,
                    "content_length": len(chunk)
                }

                # Create a Document object
                documents.append(Document(page_content=chunk, metadata=metadata))

        except Exception as e:
            print(f"Error processing {title}: {e}")

    # Batch insert documents to Supabase
    batch_size = 50  # Adjust batch size as necessary
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            SupabaseVectorStore.from_documents(
                batch,
                embedding=embeddings,
                client=supabase_client,
                table_name="documents",
                query_name="match_documents",
                chunk_size=1000
            )
            print(f"Successfully uploaded batch {i // batch_size + 1}.")
        except Exception as e:
            print(f"Error storing batch {i // batch_size + 1}: {e}")


def perform_similarity_search(query):
    """Perform a similarity search using LangChain."""
    print("Performing similarity search...")
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        table_name="documents",
        embedding=embeddings,
        query_name="match_documents"
    )
    results = vector_store.similarity_search(query)
    return results

def main():
    print("Searching for cooking-related books...")
    matching_books = search_gutenberg_titles(COOKING_KEYWORDS)
    print(f"Found {len(matching_books)} books.")

    print("Downloading and storing books...")
    download_and_store_books(matching_books)

    query = "How to bake a cake"
    print("Performing similarity search...")
    results = perform_similarity_search(query)

    for result in results:
        snippet = result.page_content[:500]  # Shortened for display
        metadata = result.metadata
        print(f"Snippet: {snippet}\nMetadata: {metadata}")
        print("-" * 50)

if __name__ == "__main__":
    main()