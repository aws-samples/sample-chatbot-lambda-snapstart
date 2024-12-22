import os
import json
import ctypes
import boto3
import llama_cpp
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global variables to persist across invocations
llm = None
model_fd = None

def create_memfd():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    MFD_CLOEXEC = 1
    
    memfd_create = libc.memfd_create
    memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    memfd_create.restype = ctypes.c_int
    
    fd = memfd_create(b"model", MFD_CLOEXEC)
    if fd == -1:
        errno = ctypes.get_errno()
        raise OSError(errno, f"memfd_create failed: {os.strerror(errno)}")
    
    return fd

def download_part(s3_client, bucket, key, fd, part):
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
    s3 = boto3.client('s3')
    
    # Get file size
    response = s3.head_object(Bucket=bucket, Key=key)
    file_size = response['ContentLength']
    
    # Create memory file
    fd = create_memfd()
    
    # Pre-allocate the full file size
    os.ftruncate(fd, file_size)
    
    # Calculate parts
    parts = []
    for start in range(0, file_size, chunk_size):
        end = min(start + chunk_size - 1, file_size - 1)
        parts.append({'start': start, 'end': end})
    
    logger.info(f"Downloading {file_size/1024/1024:.2f}MB in {len(parts)} parts")
    
    # Download parts concurrently
    download_func = partial(download_part, s3, bucket, key, fd)
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(download_func, parts)
    
    fd_path = f"/proc/self/fd/{fd}"
    return fd, fd_path

def cleanup_fd(fd):
    """Safely close the file descriptor"""
    try:
        if fd is not None:
            os.close(fd)
            logger.info(f"Successfully closed file descriptor: {fd}")
    except OSError as e:
        logger.info(f"Error closing file descriptor: {e}")
        
# Initialize model during cold start
def init_model():
    global llm, model_fd
    
    # Get model location from environment variables
    bucket = os.environ['MODEL_BUCKET']
    key = os.environ['MODEL_KEY']
    
    logger.info("Starting model download...")
    model_fd = None
    try:
        model_fd, fd_path = download_model_to_memfd(bucket, key)
        logger.info("Model download complete")
        
        logger.info("Initializing LLM...")
        llm = llama_cpp.Llama(
            model_path=fd_path,
            n_ctx=8*1024,
            n_threads=multiprocessing.cpu_count(),
            flash_attn=True,
            verbose=True,
        )
        logger.info("LLM initialization complete")
        system_prompt = "You are a knowledgable and helpful assistant"
        user_question = "Which city is the capital of France?"
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_question}<|im_end|>\n<|im_start|>assistant\n"
        
        llm.create_completion(
            prompt=prompt,
            max_tokens=6,
            temperature=0.1
        )
        logger.info("LLM is primed")
    finally:
        # Clean up the file descriptor after LLM initialization
        if model_fd is not None:
            cleanup_fd(model_fd)

# Initialize during cold start
init_model()

# Create FastAPI application
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = os.environ['MODEL_KEY']
    messages: list[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.1
    stream: bool = True  # Always true, streaming only

@app.post("/v1/chat/completions")
async def handle_chat_completion(request: ChatCompletionRequest):
    # Convert messages to prompt format using list comprehension and join
    prompt_parts = [
        f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        for msg in request.messages
    ]
    prompt_parts.append("<|im_start|>assistant\n")
    prompt = "".join(prompt_parts)
    
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=["<|im_end|>"],
        stream=True
    )
    
    async def generate():
        for chunk in response:
            completion = chunk['choices'][0]
            if completion['finish_reason'] is None:
                # Format response in OpenAI streaming format
                chunk_data = {
                    "id": "chatcmpl-" + datetime.now().strftime("%Y%m%d%H%M%S"),
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
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
                    "id": "chatcmpl-" + datetime.now().strftime("%Y%m%d%H%M%S"),
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": completion['finish_reason']
                    }]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
