-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store your books
create table
  books (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (1536) -- 1536 works for OpenAI embeddings, change if needed
  );

-- Create a function to search for books
create function match_books (
  query_embedding vector (1536),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (books.embedding <=> query_embedding) as similarity
  from books
  where metadata @> filter
  order by books.embedding <=> query_embedding;
end;
$$
SET statement_timeout TO '360s';