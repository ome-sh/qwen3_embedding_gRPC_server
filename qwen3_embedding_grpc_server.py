# License
#
# This software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0
#
# Copyright (C) 2025 Roland Kohlhuber
#
# Note: The AI model used by this software (Qwen/Qwen3-Embedding-4B) retains its original license and is not subject to the AGPL license terms.
#
# For the complete license text, see: https://www.gnu.org/licenses/agpl-3.0.html

import asyncio
import grpc
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import traceback

# These will be generated from the .proto file
import embedding_qwen_pb2
import embedding_qwen_pb2_grpc

# --- Global Model Variables ---
model = None
device = None

def load_models():
    """Load the Qwen3 embedding model and set the device."""
    global model, device
    
    print("Loading Qwen3 embedding model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    try:
        model_name = "Qwen/Qwen3-Embedding-4B"
        model_kwargs = {}
        tokenizer_kwargs = {}
        
        # Set optimizations only for CUDA devices
        if device.type == "cuda":
            model_kwargs['torch_dtype'] = torch.float16
            tokenizer_kwargs['padding_side'] = "left"
            try:
                # This will only work if flash-attn is installed
                import flash_attn
                model_kwargs['attn_implementation'] = "flash_attention_2"
                print("Attempting to load with Flash Attention 2...")
            except ImportError:
                print("Flash Attention 2 not available, using standard attention.")

        # The 'device' argument correctly handles device placement.
        # 'trust_remote_code' is needed for this model.
        # 'device_map' is incorrect for SentenceTransformer, use 'device' instead.
        model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
            trust_remote_code=True
        )
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        traceback.print_exc()
        raise
    
    print("Qwen3 embedding model loaded successfully!")
    print(f"Model dimensions: {model.get_sentence_embedding_dimension()}")


# --- Dynamic Batcher for Single Embed Requests ---
class DynamicBatcher:
    def __init__(self, model, max_batch_size=64, batch_timeout=0.05): # 50ms timeout
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue = []
        self.lock = asyncio.Lock()
        self.batch_processor_task = None
        print(f"DynamicBatcher initialized: Max Batch Size={max_batch_size}, Timeout={batch_timeout*1000}ms")

    async def _batch_processor(self):
        while True:
            await asyncio.sleep(self.batch_timeout)
            async with self.lock:
                if self.queue:
                    await self._process_queue()

    async def _process_queue(self):
        if not self.queue: return
        
        current_batch = self.queue[:]
        self.queue.clear()
        
        requests_by_prompt = {}
        for request, future in current_batch:
            prompt = request.prompt_name or "default"
            if prompt not in requests_by_prompt:
                requests_by_prompt[prompt] = {'texts': [], 'futures': []}
            requests_by_prompt[prompt]['texts'].append(request.text)
            requests_by_prompt[prompt]['futures'].append(future)

        for prompt, data in requests_by_prompt.items():
            try:
                prompt_name = prompt if prompt != "default" else None
                embeddings = self.model.encode(data['texts'], prompt_name=prompt_name)
                for i, future in enumerate(data['futures']):
                    if not future.done(): future.set_result(embeddings[i])
            except Exception as e:
                for future in data['futures']:
                    if not future.done(): future.set_exception(e)

    async def add_request(self, request):
        future = asyncio.Future()
        async with self.lock:
            self.queue.append((request, future))
            if len(self.queue) >= self.max_batch_size:
                await self._process_queue()
        return await future

    def start(self):
        self.batch_processor_task = asyncio.create_task(self._batch_processor())

    async def stop(self):
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            await self._process_queue()

# Instantiate the batcher globally
text_batcher = None

# --- gRPC Servicer Implementation ---
class EmbeddingServiceServicer(embedding_qwen_pb2_grpc.EmbeddingServiceServicer):
    
    async def Embed(self, request: embedding_qwen_pb2.EmbedRequest, context):
        try:
            embedding = await text_batcher.add_request(request)
            return embedding_qwen_pb2.EmbedResponse(
                embedding=embedding_qwen_pb2.EmbeddingVector(values=embedding.tolist()),
                model="Qwen/Qwen3-Embedding-4B",
                dimensions=len(embedding),
                prompt_name=request.prompt_name
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def EmbedBatch(self, request: embedding_qwen_pb2.TextBatchRequest, context):
        try:
            embeddings = model.encode(list(request.texts), prompt_name=request.prompt_name or None)
            return embedding_qwen_pb2.BatchEmbeddingResponse(
                embeddings=[embedding_qwen_pb2.EmbeddingVector(values=e.tolist()) for e in embeddings],
                model="Qwen/Qwen3-Embedding-4B",
                dimensions=embeddings.shape[1],
                count=len(embeddings),
                prompt_name=request.prompt_name
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def EmbedQueryDocument(self, request: embedding_qwen_pb2.QueryDocumentRequest, context):
        try:
            query_emb = model.encode(request.query, prompt_name="query")
            doc_emb = model.encode(request.document)
            sim = model.similarity(query_emb, doc_emb)[0][0]
            return embedding_qwen_pb2.QueryDocumentResponse(
                query_embedding=embedding_qwen_pb2.EmbeddingVector(values=query_emb.tolist()),
                document_embedding=embedding_qwen_pb2.EmbeddingVector(values=doc_emb.tolist()),
                similarity=sim,
                model="Qwen/Qwen3-Embedding-4B",
                dimensions=len(query_emb)
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def EmbedQueryDocumentBatch(self, request: embedding_qwen_pb2.QueryDocumentBatchRequest, context):
        try:
            q_embs = model.encode(list(request.queries), prompt_name="query")
            d_embs = model.encode(list(request.documents))
            sims = model.similarity(q_embs, d_embs).diagonal()
            return embedding_qwen_pb2.QueryDocumentBatchResponse(
                query_embeddings=[embedding_qwen_pb2.EmbeddingVector(values=e.tolist()) for e in q_embs],
                document_embeddings=[embedding_qwen_pb2.EmbeddingVector(values=e.tolist()) for e in d_embs],
                similarities=sims.tolist(),
                model="Qwen/Qwen3-Embedding-4B",
                dimensions=q_embs.shape[1],
                count=len(sims)
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def CalculateSimilarity(self, request: embedding_qwen_pb2.SimilarityRequest, context):
        try:
            embs = model.encode([request.text1, request.text2], prompt_name=request.prompt_name or None)
            sim = model.similarity(embs[0], embs[1])[0][0]
            return embedding_qwen_pb2.SimilarityResponse(
                similarity=sim,
                text1_embedding=embedding_qwen_pb2.EmbeddingVector(values=embs[0].tolist()),
                text2_embedding=embedding_qwen_pb2.EmbeddingVector(values=embs[1].tolist()),
                model="Qwen/Qwen3-Embedding-4B",
                prompt_name=request.prompt_name
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            
    async def Search(self, request: embedding_qwen_pb2.SearchRequest, context):
        try:
            top_k = request.top_k or len(request.documents)
            query_emb = model.encode(request.query, prompt_name="query")
            doc_embs = model.encode(list(request.documents))
            sims = model.similarity(query_emb, doc_embs)[0]
            
            results = sorted(
                [{"doc": doc, "sim": sim, "idx": i} for i, (doc, sim) in enumerate(zip(request.documents, sims))],
                key=lambda x: x["sim"],
                reverse=True
            )[:top_k]
            
            return embedding_qwen_pb2.SearchResponse(
                query=request.query,
                results=[embedding_qwen_pb2.SearchResultItem(document=r["doc"], similarity=r["sim"], index=r["idx"]) for r in results],
                model="Qwen/Qwen3-Embedding-4B",
                total_documents=len(request.documents)
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetInfo(self, request: embedding_qwen_pb2.InfoRequest, context):
        return embedding_qwen_pb2.InfoResponse(
            model_name="Qwen/Qwen3-Embedding-4B",
            device=str(device),
            embeddings_dimension=model.get_sentence_embedding_dimension(),
            available_prompts=list(getattr(model, 'prompts', {}).keys()),
            rpc_endpoints=[method for method in dir(EmbeddingServiceServicer) if not method.startswith('_')]
        )

# --- Server Startup ---
async def serve():
    global text_batcher
    
    # 1. Load the model first
    load_models()
    
    # 2. Initialize the batcher with the loaded model
    text_batcher = DynamicBatcher(model)
    
    # 3. Configure and start the gRPC server
    server = grpc.aio.server()
    embedding_qwen_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingServiceServicer(), server)
    
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    text_batcher.start()
    
    print("Starting Qwen3 gRPC server...")
    print(f"Listening on {listen_addr}")
    
    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        print("Shutting down server...")
        await server.stop(0)
        await text_batcher.stop()
        print("Shutdown complete.")

if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\nServer stopping...")
