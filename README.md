# Qwen3 Embedding Service

A high-performance gRPC-based embedding service using Qwen3-Embedding-4B with dynamic batching, similarity search, and query-document matching capabilities.

## Features

- **State-of-the-Art Model**: Qwen3-Embedding-4B with 4 billion parameters
- **Dynamic Batching**: Intelligent request batching for optimal throughput and latency
- **Hardware Acceleration**: CUDA/MPS/CPU support with Flash Attention 2 optimization
- **Comprehensive API**: Single/batch embeddings, similarity calculation, and semantic search
- **Query-Document Matching**: Specialized endpoints for retrieval tasks
- **Performance Optimized**: FP16 inference on CUDA, efficient memory usage

## Quick Start

### Installation

```bash
pip install grpcio grpcio-tools torch sentence-transformers
# Optional: Flash Attention 2 for CUDA optimization
pip install flash-attn
```

### Generate gRPC Files

```bash
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. embedding_qwen.proto
```

### Run the Service

```bash
python qwen_embedding_server.py
```

The service will start on `[::]:50051`

## API Endpoints

### Single Embedding
```python
# Request: EmbedRequest
{
  "text": "Your text here",
  "prompt_name": "query"  # optional
}
```

### Batch Embeddings
```python
# Request: TextBatchRequest  
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "prompt_name": "passage"  # optional
}
```

### Query-Document Matching
```python
# Request: QueryDocumentRequest
{
  "query": "What is machine learning?",
  "document": "Machine learning is a subset of artificial intelligence..."
}
```

### Similarity Calculation
```python
# Request: SimilarityRequest
{
  "text1": "First text",
  "text2": "Second text",
  "prompt_name": "default"  # optional
}
```

### Semantic Search
```python
# Request: SearchRequest
{
  "query": "Search query",
  "documents": ["Doc 1", "Doc 2", "Doc 3"],
  "top_k": 2  # optional, defaults to all documents
}
```

### Service Information
```python
# Request: InfoRequest
# Returns model info, device, dimensions, available prompts
```

## Performance Features

- **Dynamic Batching**: 64 max batch size, 50ms timeout
- **Hardware Optimization**: Automatic CUDA/MPS detection
- **Flash Attention 2**: Enhanced performance on compatible GPUs
- **FP16 Inference**: Reduced memory usage on CUDA devices
- **Pipeline Caching**: Efficient prompt-based processing

## Configuration

- **Model**: Qwen/Qwen3-Embedding-4B (4B parameters)
- **Dimensions**: Variable (check via `/GetInfo` endpoint)
- **Batch Size**: Up to 64 requests per batch
- **Timeout**: 50ms for low-latency processing
- **Port**: 50051 (default gRPC)

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (8GB+ VRAM)
- **CPU**: Multi-core processor for CPU inference
- **RAM**: 8GB+ system memory
- **Apple Silicon**: MPS support for M1/M2/M3 Macs

## Example Usage

```python
import grpc
import embedding_qwen_pb2
import embedding_qwen_pb2_grpc

# Connect to service
channel = grpc.insecure_channel('localhost:50051')
stub = embedding_qwen_pb2_grpc.EmbeddingServiceStub(channel)

# Single embedding
request = embedding_qwen_pb2.EmbedRequest(
    text="Hello world",
    prompt_name="query"
)
response = stub.Embed(request)
print(f"Embedding: {response.embedding.values[:5]}...")

# Similarity search
search_request = embedding_qwen_pb2.SearchRequest(
    query="machine learning",
    documents=["AI is fascinating", "Cooking recipes", "ML algorithms"],
    top_k=2
)
search_response = stub.Search(search_request)
for result in search_response.results:
    print(f"Doc: {result.document} - Similarity: {result.similarity:.4f}")
```

## License

AGPL-3.0 (Qwen3-Embedding-4B model retains its original license)
