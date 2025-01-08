import os
import re
import json
import argparse
from dotenv import load_dotenv

# Project Gutenberg
from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import get_text_by_id

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_community.query_constructors.supabase import SupabaseVectorTranslator
from langchain.chains.query_constructor.base import StructuredQueryOutputParser, get_query_constructor_prompt
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Supabase
from supabase import create_client, Client
from supabase.client import ClientOptions


###############################################################################
# NV & GLOBALS
###############################################################################


# Constants
# Define a list of keywords to search for in Project Gutenberg
COOKING_KEYWORDS = ["cooking", "recipes", "cookbook", "culinary"]

COMMON_INGREDIENTS = {
    "flour", "sugar", "butter", "salt", "milk", "egg", "vanilla", "baking powder",
    "baking soda", "oil", "water", "yeast", "honey", "cinnamon", "chocolate",
    "garlic", "onion", "tomato", "cheese", "beef", "chicken", "pork", "fish",
    "carrot", "potato", "pepper", "cream", "rice", "pasta", "broth", "vinegar",
    "herbs", "spices", "nuts", "almonds", "walnuts", "raisins", "yeast"
}

# Define lists of recipe types, cuisines, and special considerations

RECIPE_TYPE = ["dessert", "soup", "salad", "main course", "appetizer", "beverage"]

CUISINE = ["italian", "french", "german", "australian", "english",  "american", "thai", "japanese", "chinese", "mexican", "indian"]

SPECIAL_CONSIDERATIONS = ["vegetarian", "vegan", "keto", "nut-free", "dairy-free", "gluten-free", "low-carb"]

# Define the maximum number of tokens per chunk
MAX_TOKENS_PER_CHUNK = 128000  # approximate chunk size limit

# Define the headings that are commonly used in recipes
HEADINGS = ["INGREDIENTS", "METHOD", "INSTRUCTIONS", "DIRECTIONS"]

###############################################################################
# GUTENBERG SEARCH & METADATA
###############################################################################

def search_gutenberg_titles(cache, keywords, top_n=10, start_date=None, end_date=None):
    """
    Search Project Gutenberg for cooking-related books, optionally filtered by date.
    Returns: List of (gutenbergbookid, title).
    """
    matching_books = []
    keyword_filters = " OR ".join([f"s.name LIKE '%{kw}%'" for kw in keywords])

    date_filter = ""
    if start_date and end_date:
        date_filter = f"AND b.dateissued BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        date_filter = f"AND b.dateissued >= '{start_date}'"
    elif end_date:
        date_filter = f"AND b.dateissued <= '{end_date}'"

    query = f"""
        SELECT DISTINCT b.gutenbergbookid AS gutenbergbookid, t.name AS title
        FROM books b
        LEFT JOIN titles t ON b.id = t.bookid
        LEFT JOIN book_subjects bs ON b.id = bs.bookid
        LEFT JOIN subjects s ON bs.subjectid = s.id
        WHERE ({keyword_filters}) {date_filter}
        LIMIT {top_n};
    """
    results = cache.native_query(query)
    for row in results:
        gutenbergbookid, title = row
        matching_books.append((gutenbergbookid, title))
    return matching_books


def construct_metadata(cache, gutenberg_book_id):
    """
    Build minimal metadata from Gutenberg's cache to attach to each chunk.
    """
    query = f"""
        SELECT 
            b.gutenbergbookid AS gutenbergbookid,
            b.dateissued AS dateissued, 
            t.name AS title, 
            GROUP_CONCAT(a.name, '# ') AS authors,
            GROUP_CONCAT(s.name, '# ') AS subjects
        FROM books b
        LEFT JOIN titles t ON b.id = t.bookid
        LEFT JOIN book_authors ba ON b.id = ba.bookid
        LEFT JOIN authors a ON ba.authorid = a.id
        LEFT JOIN book_subjects bs ON b.id = bs.bookid
        LEFT JOIN subjects s ON bs.subjectid = s.id
        WHERE b.gutenbergbookid = {gutenberg_book_id}
        GROUP BY b.id, t.name;
    """
    cursor = cache.native_query(query)

    # Handle the cursor result correctly
    result = None
    for row in cursor:
        result = row

    # Ensure result exists
    if not result:
        print(f"No metadata found for book ID {gutenberg_book_id}.")
        return {
            "gutenberg_id": gutenberg_book_id,
            "title": "Unknown",
            "authors": [],
            "subjects": []
        }

    gutenberg_id, dateissued, title, authors, subjects = result
    if authors is None:
        authors = "Unknown"
    if subjects is None:
        subjects = "Unknown"

    return {
        "gutenberg_id": gutenberg_id,
        "date_issued": dateissued,
        "title": title,
        "authors": authors.split("# ") if authors else [],
        "subjects": subjects.split("# ") if subjects else [],
    }


###############################################################################
# DISCLAIMER REMOVAL
###############################################################################

def remove_gutenberg_disclaimers(book_text: str) -> str:
    """
    Removes Project Gutenberg headers/footers between
    *** START OF THIS PROJECT GUTENBERG EBOOK *** and
    *** END OF THIS PROJECT GUTENBERG EBOOK ***,
    plus lines with 'gutenberg', 'license', or URLs.
    """
    start_marker = re.search(
        r"(\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*)",
        book_text, re.IGNORECASE | re.DOTALL
    )
    end_marker = re.search(
        r"(\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*)",
        book_text, re.IGNORECASE | re.DOTALL
    )

    if start_marker and end_marker:
        start_idx = start_marker.end()
        end_idx = end_marker.start()
        book_text = book_text[start_idx:end_idx]

    filtered_lines = []
    for line in book_text.splitlines():
        lower_line = line.lower()
        if ("gutenberg" in lower_line or
            "license" in lower_line or
            "www." in lower_line or
            "http" in lower_line):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


###############################################################################
# PRE-PROCESSING FIX FOR INLINE TITLES
###############################################################################

def fix_inlined_titles(text: str) -> str:
    """
    Insert a newline after uppercase words that might be a recipe title
    followed immediately by a word like 'Makes', 'Serves', or a digit, etc.

    E.g., "BLUE CHEESE CHICKEN SPREADMakes about 40..."
    =>     "BLUE CHEESE CHICKEN SPREAD\nMakes about 40..."
    """
    pattern = re.compile(
        r'([A-Z]{2,}(?:\s+[A-Z]{2,}){0,6})(?=(Makes|Serves|\d|[^a-z\s]))'
    )

    def insert_newline(match):
        return match.group(1) + "\n"

    fixed_text = re.sub(pattern, insert_newline, text)
    return fixed_text


###############################################################################
# RECIPE DETECTION HELPERS
###############################################################################

def is_recipe_title(line: str) -> bool:
    """
    Checks if a line is likely a recipe title:
      - short (under ~15 words)
      - uppercase ratio or 'No. <digits>'
      - all words capitalized
    """
    line = line.strip()
    if not line:
        return False

    skip_words = ["chapter", "license", "project gutenberg", "www.", "http"]
    if any(sw in line.lower() for sw in skip_words):
        return False

    words = line.split()
    if len(words) == 0 or len(words) > 15:
        return False

    # e.g. "No. 211." or "Recipe No. 12"
    if re.match(r"^No\.\s*\d+", line, re.IGNORECASE):
        return True
    if re.match(r"^Recipe\s+No\.\s*\d+", line, re.IGNORECASE):
        return True

    # uppercase ratio
    alpha_chars = [c for c in line if c.isalpha()]
    if alpha_chars:
        uppercase_chars = [c for c in alpha_chars if c.isupper()]
        uppercase_ratio = len(uppercase_chars) / len(alpha_chars)
        if uppercase_ratio > 0.7:
            return True

    # all words capitalized
    capitalized_words = sum(1 for w in words if w and w[0].isupper())
    if capitalized_words == len(words):
        return True

    return False


def is_recipe_heading(paragraph: str) -> bool:
    """
    Check if a paragraph is a known heading (like "INGREDIENTS:", "METHOD:", etc.).
    We'll do a simple check: if the line is short, uppercase or ends in a colon,
    or belongs to HEADINGS.
    """
    p = paragraph.strip()
    # uppercase or ends with a colon
    if p.isupper() or p.endswith(":"):
        return True
    # check HEADINGS
    for h in HEADINGS:
        if h in p.upper():
            return True
    return False


###############################################################################
# EXTRACTION WITH "OVERSAMPLING" & MULTIPLE TITLES
###############################################################################

def extract_all_recipes_with_context(book_text: str, oversample=1):
    """
    Extract recipes by:
      - Splitting text into paragraphs
      - Searching for lines that look like recipe titles
      - When a new title is found, finalize the previous recipe and start a new one
      - "Oversample" = also include N paragraphs before a recognized title
      - If multiple titles occur in the same paragraph, handle them separately
      - If we see headings like "INGREDIENTS" or "METHOD" in the same paragraph,
        we keep them with the current recipe.

    Return a list of text chunks, each chunk representing a single recipe + context.
    """

    raw_paragraphs = book_text.split("\n\n")
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    expanded_paragraphs = []
    for para in paragraphs:
        lines = para.splitlines()
        tmp_buffer = []
        for line in lines:
            line_stripped = line.strip()
            if is_recipe_title(line_stripped):
                if tmp_buffer:
                    expanded_paragraphs.append("\n".join(tmp_buffer).strip())
                    tmp_buffer = []
                expanded_paragraphs.append(line_stripped)
            else:
                tmp_buffer.append(line_stripped)
        if tmp_buffer:
            expanded_paragraphs.append("\n".join(tmp_buffer).strip())

    recipes = []
    current_recipe = []
    in_recipe = False

    i = 0
    while i < len(expanded_paragraphs):
        p = expanded_paragraphs[i].strip()

        if is_recipe_title(p):
            # finalize old recipe
            if in_recipe and current_recipe:
                recipes.append("\n\n".join(current_recipe))
                current_recipe = []

            # start new recipe
            in_recipe = True

            # oversample: also include up to `oversample` paragraphs before this one
            start_idx = max(0, i - oversample)
            for back_idx in range(start_idx, i):
                if expanded_paragraphs[back_idx] not in current_recipe:
                    current_recipe.append(expanded_paragraphs[back_idx])
            current_recipe.append(p)

        else:
            if in_recipe:
                if is_recipe_heading(p):
                    current_recipe.append(p)
                else:
                    current_recipe.append(p)

        i += 1

    # finalize last recipe
    if in_recipe and current_recipe:
        recipes.append("\n\n".join(current_recipe))

    return recipes


###############################################################################
# LLM-BASED RECIPE PARSING (Single Call for Full Metadata)
###############################################################################

def extract_recipe_info(chunk_text: str, llm: ChatOpenAI) -> dict:
    """
    Single call to the LLM that extracts:
      - recipe_found (bool)
      - title (str)
      - ingredients (list of str)
      - instructions (str)
      - recipe_type (str or list[str])
      - cuisine (str)
      - special_considerations (str or list[str])

    If not a recipe, returns {"recipe_found": false}.
    """
    system_prompt = (
        "You are a helpful assistant that identifies and extracts recipes from text. "
        "Return your answer in valid JSON. If no recipe is present, return "
        '{"recipe_found": false}.\n\n'
        "If a recipe is found, return a JSON object with:\n"
        "{\n"
        '  "recipe_found": true,\n'
        '  "title": "STRING with recipe title",\n'
        '  "ingredients": ["LIST", "OF", "INGREDIENTS", "in", "lowercase", "without", "quantities"],\n'
        '  "instructions": "STRING with instructions",\n'
        '  f"recipe_type": "STRING or LIST OF STRINGS consisting stricly of one or more of the following: {RECIPE_TYPE}",\n'
        '  f"cuisine": "STRING like the following: {CUISINE}",\n'
        '  f"special_considerations": "STRING or LIST OF STRINGS consisting of one or more of the following special dietary considerations: {SPECIAL_CONSIDERATIONS}""\n'
        "}\n"
        "Do not add fields beyond these. The entire reply must be valid JSON only."
    )

    user_prompt = (
        f"Text chunk:\n{chunk_text}\n\n"
        "Does this text contain a recipe? If yes, extract all relevant data above. "
        "If no recipe is present, return {\"recipe_found\": false}."
    )

    try:
                # Create message objects
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Invoke the LLM with the messages
        response: AIMessage = llm.invoke(messages)

        reply = response.content.strip()
        recipe_data = json.loads(reply)
        print(f"\nChunk: {chunk_text}")
        print("\n************\n")
        print(f"LLM reply: {recipe_data} \n")
        return recipe_data
    except Exception as e:
        print(f"LLM parsing error: {e}")
        return {"recipe_found": False}


###############################################################################
# TOKEN COUNT & VARIABLE-SIZED CHUNKING
###############################################################################

def approximate_token_count(text: str) -> int:
    """
    Rough token estimate (words * 1.3).
    """
    words = text.split()
    return int(len(words) * 1.3)


###############################################################################
# DOWNLOAD, EXTRACT, OVERSAMPLE, LLM-VALIDATE, & STORE
###############################################################################

def download_and_store_books(matching_books, cache, llm, vector_store, oversample=1):
    """
    Pipeline:
      1. Download text
      2. Remove disclaimers
      3. Fix inline titles
      4. Extract recipes with oversampling
      5. If large, subdivide the chunk
      6. Single LLM call that returns all metadata: 
         (title, instructions, ingredients, recipe_type, cuisine, special_considerations)
      7. Store the recognized recipe in Supabase
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_TOKENS_PER_CHUNK, chunk_overlap=200)
    documents = []

    for gutenberg_book_id, title in matching_books:
        print(f"Processing: {title} (ID: {gutenberg_book_id})")

        try:
            # Build metadata
            metadata = construct_metadata(cache, gutenberg_book_id)

            # Get raw text
            raw_text = get_text_by_id(gutenberg_book_id)
            if not raw_text:
                print(f"Unable to retrieve content for ID {gutenberg_book_id}.")
                continue

            # Convert to string
            content = raw_text.decode("utf-8", errors="ignore")

            # Remove Gutenberg disclaimers
            content = remove_gutenberg_disclaimers(content)

            # Fix inline all-caps titles
            content = fix_inlined_titles(content)

            # Extract 'recipes' with oversampling
            recipe_texts = extract_all_recipes_with_context(content, oversample=oversample)

            # For each extracted recipe:
            for i, recipe_text in enumerate(recipe_texts):
                token_count = approximate_token_count(recipe_text)

                # If big, subdivide
                if token_count <= MAX_TOKENS_PER_CHUNK:
                    sub_chunks = [recipe_text]
                else:
                    sub_chunks = text_splitter.split_text(recipe_text)

                for j, sub_chunk in enumerate(sub_chunks):
                    # 5) Single LLM call for full metadata
                    recipe_info = extract_recipe_info(sub_chunk, llm)
                    if recipe_info.get("recipe_found"):
                        # Merge chunk-level info with general Gutenberg metadata
                        chunk_metadata = metadata.copy()
                        chunk_metadata["recipe_index"] = i
                        chunk_metadata["sub_chunk_index"] = j
                        chunk_metadata["token_count"] = approximate_token_count(sub_chunk)

                        # === Attach the LLM-extracted fields ===
                        # If the LLM fails to parse a specific field, default to empty
                        chunk_metadata["recipe_title"] = recipe_info.get("title", "")
                        chunk_metadata["ingredients"] = recipe_info.get("ingredients", [])
                        # chunk_metadata["instructions"] = recipe_info.get("instructions", "")
                        chunk_metadata["recipe_type"] = recipe_info.get("recipe_type", "")
                        chunk_metadata["cuisine"] = recipe_info.get("cuisine", "")
                        chunk_metadata["special_considerations"] = recipe_info.get("special_considerations", "")

                        formatted_recipe = f"Title: {recipe_info.get('title')}\n\n" \
                                             f"Ingredients: {recipe_info.get('ingredients')}\n\n" \
                                                f"Instructions: {recipe_info.get('instructions')}\n\n"
                        
                        # === Create a Document with the final text + metadata ===
                        document = Document(page_content=formatted_recipe, metadata=chunk_metadata)
                        documents.append(document)

        except Exception as e:
            print(f"Error processing {title} (ID: {gutenberg_book_id}): {e}")

    # Batch upload recognized recipes to Supabase
    batch_size = 50 # Adjust as necessary
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        try:
            vector_store.add_documents(
                batch
            )
            print(f"Successfully uploaded batch {i//batch_size + 1} "
                  f"of {len(documents)//batch_size + 1}.")
        except Exception as e:
            print(f"Error storing batch {i//batch_size + 1}: {e}")


###############################################################################
# SELF-QUERY RETRIEVER
###############################################################################


def perform_self_query_retrieval(query, llm, vector_store, structured_query_translator):
    """
    Creates a SelfQueryRetriever for the following metadata fields:
      - recipe_title
      - recipe_type
      - cuisine
      - special_considerations
      - ingredients
    """

    metadata_field_info = [
        AttributeInfo(
            name="recipe_title",
            description="The title of the recipe. Use the like operator for partial matches.",
            type="string",
        ),
        AttributeInfo(
            name="recipe_type",
            description=f"The type of recipe (e.g., {RECIPE_TYPE}).",
            type="string",
        ),
        AttributeInfo(
            name="cuisine",
            description=f"The cuisine type (e.g., {CUISINE}). Use the like operator for partial matches.",
            type="string",
        ),
        AttributeInfo(
            name="special_considerations",
            description=f"Dietary restrictions (e.g., {SPECIAL_CONSIDERATIONS}). Use the like operator for partial matches.",
            type="list[string]",
        ),
        AttributeInfo(
            name="ingredients",
            description=f"Key ingredients in the recipe (e.g., {COMMON_INGREDIENTS}). Use the like operator for partial matches.",
            type="list[string]",
        ),
    ]

    doc_content_desc = "Text content describing a cooking recipe"
    doc_contents = (
        "The text content of a cooking recipe, including its ingredients, "
        "instructions, and relevant metadata."
    )
    
    examples = [
        (
            "Show me all American dessert recipes but not vegetarian.",
            {
                "query": "American dessert",
                "filter": """and(
                                eq("cuisine", 'american'), 
                                eq("recipe_type", 'dessert'), 
                                ne("special_considerations", 'vegetarian')
                            )"""
            }
        ),
        (
            "Show me all vegetarian or vegan recipes that include both rice and broccoli.",
            {
                "query": "rice broccoli",
                "filter": """and(
                                or(
                                    like("special_considerations", '%vegetarian%'), 
                                    like("special_considerations", '%vegan%')
                                ), 
                                like("ingredients", '%rice%'), 
                                like("ingredients", '%broccoli%')
                            )"""
            }
        ),
        (
            "Show me all Italian recipes that are not desserts and include chocolate.",
            {
                "query": "Italian chocolate",
                "filter": """and(
                                eq("cuisine", 'italian'), 
                                ne("recipe_type", 'dessert'), 
                                like("ingredients", '%chocolate%')
                            )"""
            }
        ),
        (
            "Show me all recipes that are either American or Italian and are low-carb desserts.",
            {
                "query": "low-carb dessert",
                "filter": """and(
                                or(
                                    eq("cuisine", 'american'), 
                                    eq("cuisine", 'italian')
                                ), 
                                eq("recipe_type", 'dessert'), 
                                like("special_considerations", '%low-carb%')
                            )"""
            }
        ),
        (
            "Show me all recipes that are not desserts, contain chicken, and are gluten-free or dairy-free.",
            {
                "query": "chicken",
                "filter": """and(
                                ne("recipe_type", 'dessert'), 
                                like("ingredients", '%chicken%'), 
                                or(
                                    like("special_considerations", '%gluten-free%'), 
                                    like("special_considerations", '%dairy-free%')
                                )
                            )"""
            }
        ),
        (
            "Show me all vegan recipes that are either Italian or Mexican and include chocolate and vanilla.",
            {
                "query": "vegan chocolate vanilla",
                "filter": """and(
                                like("special_considerations", '%vegan%'), 
                                or(
                                    eq("cuisine", 'italian'), 
                                    eq("cuisine", 'mexican')
                                ), 
                                like("ingredients", '%chocolate%'), 
                                like("ingredients", '%vanilla%')
                            )"""
            }
        ),
        (
            "Show me all recipes with 'cake' in the title that are American but not low-carb.",
            {
                "query": "cake",
                "filter": """and(
                                like("recipe_title", '%cake%'), 
                                eq("cuisine", 'american'), 
                                not(like("special_considerations", '%low-carb%'))
                            )"""
            }
        ),
    ]

    prompt = get_query_constructor_prompt(
        doc_content_desc,
        metadata_field_info,
        examples=examples
    )

    # print(prompt.format(query="Show me chocolate cake recipes."))

    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | llm | output_parser

    sq_retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vector_store,
    structured_query_translator=structured_query_translator,
    )

    # Output parser will split the LLM result into a list of queries
    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines))  # Remove empty lines


    output_parser = LineListOutputParser()

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Chain
    mq_chain = query_prompt | llm | output_parser

    mq_retriever = MultiQueryRetriever(
        retriever=sq_retriever, llm_chain=mq_chain, parser_key="lines"
    )

    recipes = mq_retriever.invoke(query)

    chain = RunnableParallel(nutrition=generate_nutrition_info_chain(llm), shopping_list=generate_shopping_list_chain(llm), factoids=generate_factoids_chain(llm), recipe=RunnablePassthrough())

    outputs = []

    for i, recipe in enumerate(recipes, start=1):
        output = chain.invoke({"text": recipe.page_content, "metadata": recipe.metadata })

        # Extract raw text output from the LLM
        processed_output = {
            "nutrition": output["nutrition"],
            "shopping_list": output["shopping_list"],
            "factoids": output["factoids"],
            "recipe": output["recipe"]  # Recipe is already raw text
        }
        outputs.append(processed_output)
        

    return outputs



###############################################################################
# LLM CHAINS
###############################################################################

def generate_nutrition_info_chain(llm):
    """
    Calculate estimated calories and macronutrients for a list of ingredients.
    """
    return ChatPromptTemplate.from_template (
        """You are a nutrition assistant. Given a list of ingredients, estimate the total
        calories, protein, carbs, and fat. Return only a single valid JSON object in the following format.
        
        {{
            "description": "nutrition info",
            "ingredients": [
                {{"name": "Honey", "amount": "1/2 lb", "calories": 690, "protein": 0.3, "carbs": 177, "fat": 0}},
                {{"name": "Almonds", "amount": "1/2 lb", "calories": 650, "protein": 24, "carbs": 23, "fat": 56}},
                {{"name": "Filberts", "amount": "1/2 lb", "calories": 700, "protein": 15, "carbs": 16, "fat": 66}},
                {{"name": "Candied Lemon Peel", "amount": "1/4 cup", "calories": 120, "protein": 0, "carbs": 30, "fat": 0}},
                {{"name": "Pepper", "amount": "1 tsp", "calories": 6, "protein": 0.2, "carbs": 1.4, "fat": 0.1}},
                {{"name": "Cinnamon", "amount": "1 tsp", "calories": 6, "protein": 0.1, "carbs": 2, "fat": 0.03}},
                {{"name": "Chocolate", "amount": "1/4 lb", "calories": 600, "protein": 7, "carbs": 50, "fat": 45}},
                {{"name": "Corn Flour", "amount": "1 tbsp", "calories": 30, "protein": 1, "carbs": 7, "fat": 0.1}},
                {{"name": "Large Wafers", "amount": "4 wafers", "calories": 80, "protein": 1, "carbs": 16, "fat": 2}}
            ],
            "total": {{
                "calories": 2882,
                "protein": 49.6,
                "carbs": 302.4,
                "fat": 169.33
            }}
        }}

        \n
        {text}"""
    ) | llm | StrOutputParser()

def generate_shopping_list_chain(llm):
    """
    Generate a shopping list from the given ingredients.
    """
    return ChatPromptTemplate.from_template (
        """You are a shopping assistant. Analyze the recipe text and create a shopping list of items needed for this recipe.
          Return only a single valid JSON object in the following format.\n
            {{
                "description": "shopping list",
                "ingredients": [
                    "Honey",
                    "Almonds",
                    "Filberts",
                    "Candied lemon peel",
                    "Pepper",
                    "Cinnamon",
                    "Chocolate",
                    "Corn flour",
                    "Large wafers"
                ]
            }}
        \n
        {text}"""
    ) | llm | StrOutputParser()

def generate_factoids_chain(llm):
    """
    Generate interesting factoids about the recipe's ingredients and methods.
    """
    return ChatPromptTemplate.from_template (
        """You are a culinary historian. Analyze the recipe text and provide interesting
        factoids about its ingredients and methods. Return only a single valid JSON object in the following format.\n
        
        {{
            "description": "Factoids for ingredients and methods",
            "data": [
                {{"type": "ingredient", "name": "Honey", "fact": "Honey has been used as a sweetener for thousands of years and was highly valued in ancient cultures, often associated with immortality and used in religious rituals."}},
                {{"type": "ingredient", "name": "Almonds", "fact": "Almonds are one of the oldest cultivated nuts, with origins tracing back to the Middle East and South Asia. They are rich in healthy fats, fiber, and protein."}},
                {{"type": "ingredient", "name": "Filberts", "fact": "Filberts, also known as hazelnuts, have been cultivated since the Bronze Age and are often associated with fertility and protection in various cultures."}},
                {{"type": "ingredient", "name": "Candied lemon peel", "fact": "Candied citrus peels have been used in European confections since the Middle Ages, serving as a way to preserve fruit and add flavor to sweets."}},
                {{"type": "ingredient", "name": "Pepper", "fact": "Black pepper was once so valuable that it was used as currency in trade, and it has been a staple spice in cooking for over 4,000 years."}},
                {{"type": "ingredient", "name": "Cinnamon", "fact": "Cinnamon is one of the oldest known spices, with a history of use dating back to ancient Egypt, where it was highly prized and used in embalming."}},
                {{"type": "ingredient", "name": "Chocolate", "fact": "Chocolate originated from the cacao bean, which was used by ancient Mesoamerican cultures in ceremonial drinks. It was introduced to Europe in the 16th century."}},
                {{"type": "ingredient", "name": "Corn flour", "fact": "Corn flour is a staple in many cuisines, particularly in the Americas, and is used to thicken sauces and batters due to its fine texture."}},
                {{"type": "ingredient", "name": "Large wafers", "fact": "Wafers have a long history in baking, often used as a base for confections and desserts. They can be traced back to ancient Greece and Rome."}},
                {{"type": "method", "name": "Boiling honey in a copper vessel", "fact": "Copper vessels are traditionally used in candy making because they conduct heat evenly, allowing for precise temperature control, which is crucial for achieving the right texture."}},
                {{"type": "method", "name": "Baking in a slow oven", "fact": "Baking at low temperatures allows for gradual cooking, which helps to develop flavors and maintain moisture in baked goods, a technique often used in traditional recipes."}}
            ]
        }}
        \n
        {text}"""
    ) | llm | StrOutputParser()

###############################################################################
# MAIN
###############################################################################


def main():

    parser = argparse.ArgumentParser(
        description="Loading and testing a vector store."
    )
    
    parser.add_argument("-lb", "--load_books", type=bool, default=False, help="Search and load books.")
    
    parser.add_argument("-n", "--top_n", type=int, default=3, help="Number of books to load.")
    
    parser.add_argument("-sd", "--start_date", type=str, default="1950-01-01", help="Search start date.")

    parser.add_argument("-ed", "--end_date", type=str, default="2000-12-31", help="Search end date.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    top_n = args.top_n
    start_date = args.start_date
    end_date = args.end_date

    # Load environment variables
    load_dotenv(override=True) # Load environment variables from .env

    SUPABASE_URL = os.getenv("SUPABASE_HTTPS_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Initialize Supabase
    supabase_client: Client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        options=ClientOptions(
            postgrest_client_timeout=360,
            storage_client_timeout=360,
            schema="public"
        )
    )

    # Initialize embeddings & LLMs
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    chat_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    classifier_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="recipes",
        query_name="match_recipes"
    )

    # Initialize Gutenberg cache
    cache = GutenbergCache.get_cache()

    if args.load_books:
        print("Searching for cooking-related books...")
        matching_books = search_gutenberg_titles(
            cache,
            keywords=COOKING_KEYWORDS,
            top_n=top_n,
            start_date=start_date,
            end_date=end_date
        )
        print(f"Found {len(matching_books)} books.")

        # Download, oversample paragraphs by 1 on each side for context
        print("Downloading and storing books...")
        oversample_distance = 1
        download_and_store_books(matching_books, cache, classifier_llm, vector_store, oversample=oversample_distance)

    # Perform query
    query = "Show me dessert recipes that are either Italian or English."
    print(f"\nSelf-query retrieval with: {query}")
    results = perform_self_query_retrieval(query, chat_llm, vector_store, SupabaseVectorTranslator())

    for i, res in enumerate(results, start=1):
        print(f"\n[Result {i}] Recipe: {res['recipe']['text']}")
        print(f"[Metadata] {res['recipe']['metadata']}")
        print(f"[Nutrition] {res['nutrition']}")
        print(f"[Shopping List] {res['shopping_list']}")
        print(f"[Factoids] {res['factoids']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
