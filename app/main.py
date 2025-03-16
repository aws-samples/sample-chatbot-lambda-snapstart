import os
import json
import ctypes
import boto3
import llama_cpp
import logging
import multiprocessing
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Global variables to persist across invocations
llm = None  # The LLM instance that will be reused across requests
model_fd = None  # File descriptor for the model file

def create_memfd():
    """Create a memory file descriptor using the memfd_create syscall.
    
    This allows us to create a file that exists only in memory, not on disk.
    
    Returns:
        int: File descriptor for the memory file
        
    Raises:
        OSError: If memfd_create fails
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    MFD_CLOEXEC = 1  # Close the fd when executing a new program
    
    memfd_create = libc.memfd_create
    memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    memfd_create.restype = ctypes.c_int
    
    fd = memfd_create(b"model", MFD_CLOEXEC)
    if fd == -1:
        errno = ctypes.get_errno()
        raise OSError(errno, f"memfd_create failed: {os.strerror(errno)}")
    
    return fd

def download_part(s3_client, bucket, key, fd, part):
    """Download a specific byte range of a file from S3 to a file descriptor.
    
    Args:
        s3_client: Boto3 S3 client
        bucket (str): S3 bucket name
        key (str): S3 object key
        fd (int): File descriptor to write to
        part (dict): Dictionary with 'start' and 'end' byte positions
    """
    start_byte = part['start']
    end_byte = part['end']
    
    # Download this part of the file
    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
        Range=f'bytes={start_byte}-{end_byte}'
    )
    
    # Write to the specific position in memfd
    data = response['Body'].read()
    os.lseek(fd, start_byte, os.SEEK_SET)
    os.write(fd, data)

def download_model_to_memfd(bucket, key, chunk_size=100*1024*1024):  # 100MB chunks
    """Download a model file from S3 to a memory file descriptor in parallel chunks.
    
    This function:
    1. Gets the file size from S3
    2. Creates a memory file descriptor
    3. Pre-allocates the full file size
    4. Splits the download into chunks
    5. Downloads chunks in parallel
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        chunk_size (int): Size of each download chunk in bytes
        
    Returns:
        tuple: (file descriptor, file path)
    """
    s3 = boto3.client('s3')
    
    # Get file size
    response = s3.head_object(Bucket=bucket, Key=key)
    file_size = response['ContentLength']
    
    # Create memory file
    fd = create_memfd()
    
    # Pre-allocate the full file size
    os.ftruncate(fd, file_size)
    
    # Calculate parts for parallel download
    parts = []
    for start in range(0, file_size, chunk_size):
        end = min(start + chunk_size - 1, file_size - 1)
        parts.append({'start': start, 'end': end})
    
    logger.info(f"Downloading {file_size/1024/1024:.2f}MB in {len(parts)} parts")
    
    # Download parts concurrently using ThreadPoolExecutor
    download_func = partial(download_part, s3, bucket, key, fd)
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(download_func, parts)
    
    # Create a path that can be used to access the file
    fd_path = f"/proc/self/fd/{fd}"
    return fd, fd_path

def cleanup_fd(fd):
    """Safely close the file descriptor.
    
    Args:
        fd (int): File descriptor to close
    """
    try:
        if fd is not None:
            os.close(fd)
            logger.info(f"Successfully closed file descriptor: {fd}")
    except OSError as e:
        logger.info(f"Error closing file descriptor: {e}")
        
# Initialize model during cold start
def init_model():
    """Initialize the LLM model during Lambda cold start.
    
    This function:
    1. Downloads the model from S3 to a memory file descriptor
    2. Initializes the LLM with the model
    3. Primes the model with a simple query to reduce latency for the first request
    4. Cleans up the file descriptor (llama-cpp-python loads the entire model into memory)
    """
    global llm, model_fd
    
    # Get model location from environment variables
    bucket = os.environ['MODEL_BUCKET']
    key = os.environ['MODEL_KEY']
    
    logger.info("Starting model download...")
    model_fd = None
    try:
        # Download model to memory file descriptor
        model_fd, fd_path = download_model_to_memfd(bucket, key)
        logger.info("Model download complete")
        
        # Initialize the LLM with the model
        logger.info("Initializing LLM...")
        llm = llama_cpp.Llama(
            model_path=fd_path,
            n_ctx=32768,           # -c 32768: Context window size
            n_batch=2048,          # -b 2048: Batch size
            n_ubatch=512,          # -ub 512: Update batch size
            n_threads=multiprocessing.cpu_count(),  # Use all available CPU cores
            flash_attn=True,       # -fa: Enable flash attention
            verbose=True,
            cache_type_k="q8_0",   # --cache-type-k q8_0: KV cache type for keys
            cache_type_v="f16",    # --cache-type-v f16: KV cache type for values
        )
        logger.info("LLM initialization complete")
        
        # Prime the model with a simple query to reduce latency for the first request
        user_question = "Which city is the capital of France?"
        prompt = f"<|User|>{user_question}<|Assistant|>"
        
        llm.create_completion(
            prompt=prompt,
            max_tokens=6,
            temperature=0.1
        )
        logger.info("LLM is primed")
    finally:
        # Clean up the file descriptor after LLM initialization
        # This is safe because llama-cpp-python loads the entire model into memory
        if model_fd is not None:
            cleanup_fd(model_fd)

# Initialize during cold start
init_model()

# Create FastAPI application
app = FastAPI()

@app.get("/healthz")
def healthz():
    """Health check endpoint for AWS Lambda Web Adapter.
    
    Returns:
        dict: Status message
    """
    return {"status": "ok"}

class Message(BaseModel):
    """Message model for chat completion requests.
    
    Attributes:
        role (str): The role of the message sender (e.g., "user", "assistant")
        content (str): The content of the message
    """
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """Request model for chat completion API.
    
    Attributes:
        model (str): The model to use for completion
        messages (list[Message]): The conversation history
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (0.0 to 1.0)
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p sampling parameter
        repeat_penalty (float): Penalty for repeated tokens
        stream (bool): Whether to stream the response
    """
    model: Optional[str] = os.environ['MODEL_KEY']
    messages: list[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    repeat_penalty: Optional[float] = 1.1
    stream: bool = True  # Always true, streaming only

@app.post("/v1/chat/completions")
async def handle_chat_completion(request: ChatCompletionRequest):
    """Handle chat completion requests in OpenAI-compatible format.
    
    This endpoint:
    1. Formats the conversation history into a prompt for the LLM
    2. Generates a completion using the LLM
    3. Streams the response back to the client in OpenAI-compatible format
    
    Args:
        request (ChatCompletionRequest): The chat completion request
        
    Returns:
        StreamingResponse: Server-sent events stream with completion chunks
    """
    # Generate a unique ID for this completion
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    created_timestamp = int(datetime.now().timestamp())

    # Convert messages to prompt format using list comprehension and join
    # Format: <|role|>content<|role|>content...
    prompt_parts = [
        f"<|{msg.role}|>{msg.content}"
        for msg in request.messages
    ]
    prompt_parts.append("<|Assistant|>\n<think>\n")  # Add the assistant prefix for the response, and force thinking with '<think>' tag
    prompt = "".join(prompt_parts)
    
    # Generate completion with streaming enabled
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repeat_penalty=request.repeat_penalty,
        # Stop generation when these strings are encountered
        stop=["<|user|>","<|assistant|>","<|User|>","<|Assistant|>", "</|assistant>"],
        stream=True
    )
    
    async def generate():
        """Generate streaming response chunks in OpenAI-compatible format.
        
        This is an async generator that:
        1. Iterates through the LLM response chunks
        2. Formats each chunk in OpenAI-compatible format
        3. Yields the formatted chunks as server-sent events
        """
        for chunk in response:
            completion = chunk['choices'][0]
            if completion['finish_reason'] is None:
                # Format response in OpenAI streaming format
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": request.model,
                    "choices": [{
                        "delta": {"content": completion['text']},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            else:
                # Send final chunk with finish_reason
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": request.model,
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": completion['finish_reason']
                    }]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
    
    # Return a streaming response with the appropriate content type
    return StreamingResponse(generate(), media_type="text/event-stream")
