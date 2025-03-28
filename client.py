import requests
import json
import argparse
import os
import readline
import sys
import time
import threading
import signal
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

# Fixed endpoint path
API_ENDPOINT = "/v1/chat/completions"

class ThinkingIndicator:
    """Animated thinking indicator for the chat client."""
    
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        
    def start(self):
        """Start the thinking animation in a separate thread."""
        print("\n", end='', flush=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate)
        self._thread.daemon = True
        self._thread.start()
        
    def stop(self):
        """Stop the thinking animation and clean up."""
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            # Clear the thinking indicator
            print("\r\033[K", end='', flush=True)
            
    def _animate(self):
        """Animation loop for the thinking indicator."""
        while not self._stop_event.is_set():
            for frame in ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]:
                if self._stop_event.is_set():
                    break
                print(f"\r\033[K{frame} Thinking...", end='', flush=True)
                time.sleep(0.1)

class ChatClient:
    """Client for interacting with the serverless LLM API."""
    
    def __init__(self, api_base: str, max_tokens: int = 512, temperature: float = 0.6,
                 credentials: Optional[Any] = None, region: Optional[str] = None):
        """Initialize the chat client.
        
        Args:
            api_base: Base URL for the API
            max_tokens: Maximum tokens to generate in responses
            temperature: Sampling temperature for text generation
            credentials: AWS credentials (optional, will be loaded from environment if None)
            region: AWS region (optional, will be extracted from URL if None)
        """
        self.api_base = api_base.rstrip('/')
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = requests.Session()
        self.messages: List[Dict[str, str]] = []
        self.thinking = ThinkingIndicator()
        self.interrupt_flag = False
        self.exit_flag = False
        self.last_interrupt_time = 0
        
        # Get AWS credentials if not provided
        if credentials is None:
            session = boto3.Session()
            self.credentials = session.get_credentials()
            if not self.credentials:
                raise ValueError("AWS credentials not found. Please configure AWS credentials.")
        else:
            self.credentials = credentials
            
        # Extract region from URL if not provided
        if region is None:
            self.region = urlparse(self.api_base).hostname.split('.')[2]
        else:
            self.region = region
            
        # Initialize conversation
        self.new_conversation()
    
    def new_conversation(self):
        """Start a new conversation by resetting the message history."""
        self.messages = []
    
    def _sign_request(self, url: str, method: str, body: Optional[Dict]) -> requests.PreparedRequest:
        """Sign an HTTP request with AWS SigV4 authentication.
        
        Args:
            url: The full URL for the request
            method: HTTP method (e.g., 'GET', 'POST')
            body: Request body to be JSON serialized
            
        Returns:
            A signed request ready to be sent
        """
        # Create an AWS request object
        aws_request = AWSRequest(
            method=method,
            url=url,
            data=json.dumps(body) if body else None,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            }
        )
        
        # Sign the request with SigV4
        SigV4Auth(self.credentials, 'lambda', self.region).add_auth(aws_request)
        
        # Convert AWSRequest to a requests-compatible format
        return requests.Request(
            method,
            url,
            headers=dict(aws_request.headers),
            data=aws_request.data
        ).prepare()
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C (SIGINT) during response generation."""
        current_time = time.time()
        
        # If this is the first interrupt or more than 1 second has passed since the last one
        if self.last_interrupt_time == 0 or (current_time - self.last_interrupt_time) > 1:
            self.interrupt_flag = True
            self.last_interrupt_time = current_time
            print("\n\n[Response interrupted. Press Ctrl+C again within 1 second to exit]")
        else:
            # This is a double interrupt within 1 second
            self.exit_flag = True
            print("\n\nExiting...")
            sys.exit(0)
    
    def send_message(self, message: str) -> Optional[str]:
        """Send a message to the LLM and get the response.
        
        Args:
            message: The user's message
            
        Returns:
            The assistant's response, or None if there was an error
        """
        # Add message to history
        self.messages.append({"role": "user", "content": message})
        
        # Reset interrupt flags
        self.interrupt_flag = False
        self.last_interrupt_time = 0
        
        # Set up signal handler for Ctrl+C
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            url = f"{self.api_base}{API_ENDPOINT}"
            body = {
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True
            }
            
            # Sign and prepare the request
            prepared_request = self._sign_request(
                url=url,
                method='POST',
                body=body
            )
            
            # Start thinking animation
            self.thinking.start()
            
            # Send the signed request with streaming
            response = self.session.send(prepared_request, stream=True)
            response.raise_for_status()
            
            # Process the streaming response
            assistant_message = ""
            first_chunk = True
            for line in response.iter_lines():
                # Check if user interrupted
                if self.interrupt_flag:
                    break
                    
                if line:
                    line = line.decode('utf-8')
                    if line == "data: [DONE]":
                        break
                        
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[6:])
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                choice = json_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    
                                    # Stop thinking animation only when first content chunk arrives
                                    if first_chunk:
                                        self.thinking.stop()
                                        print("<think>", flush=True)
                                        first_chunk = False
                                        
                                    print(content, end='', flush=True)
                                    assistant_message += content
                        except json.JSONDecodeError:
                            continue
            
            print()  # Add a newline at the end
            
            # Add successful response to history (even if interrupted)
            if assistant_message:
                if self.interrupt_flag:
                    # Add a note that the response was interrupted
                    assistant_message += " [Response interrupted by user]"
                self.messages.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            self.thinking.stop()
            print(f"\nError making request: {str(e)}")
            return None
        except Exception as e:
            if not self.exit_flag:  # Don't show error if exiting intentionally
                self.thinking.stop()
                print(f"\nUnexpected error: {str(e)}")
            return None
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

def get_api_base() -> str:
    """Get the API base URL from command line arguments or environment variables.
    
    Returns:
        The API base URL
        
    Raises:
        ValueError: If the API base URL is not provided
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive chat client')
    parser.add_argument('--api-base', help='API base URL')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature (0.0-1.0)')
    parser.add_argument('--max-tokens', type=int, default=32768,
                       help='Maximum tokens to generate')
    args = parser.parse_args()
    
    # Priority order: command line > environment variable
    if args.api_base:
        return args.api_base, args.temperature, args.max_tokens
    elif 'CHAT_API_BASE' in os.environ:
        return os.environ['CHAT_API_BASE'], args.temperature, args.max_tokens
    else:
        raise ValueError(
            "API base URL not found. Please set it via --api-base argument "
            "or CHAT_API_BASE environment variable in .env file"
        )

def main():
    """Main function for the chat client."""
    try:
        # Get configuration
        api_base, temperature, max_tokens = get_api_base()
        
        # Create chat client
        client = ChatClient(
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        print(f"Using API base URL: {api_base}")
        print(f"Temperature: {temperature}")
        print(f"Max tokens: {max_tokens}")
        print("\nChat started. Available commands:")
        print("  /quit - Exit the chat")
        print("  /new  - Start a new conversation")
        print("  Ctrl+C - Interrupt current response")
        print("  Ctrl+C twice - Exit the chat")
        print("  Use ↑/↓ keys to navigate through history")
        print("\nMulti-line input:")
        print("  Type 'EOF' and press Enter to start multi-line mode")
        print("  Type '```' and press Enter to start multi-line mode")
        print("  Then type your multi-line text and end with the same delimiter")
        
        # Configure readline for history
        histfile = ".chat_history"
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
            
        # Define a function to handle the prompt with Unicode characters
        def get_input():
            """Get user input, supporting both single-line and multi-line input with delimiters."""
            # Display primary prompt
            print("\n➤ ", end="", flush=True)
            
            # Read first line
            first_line = input().strip()
            
            # Check for multi-line delimiters
            delimiters = ["EOF", "```"]
            active_delimiter = None
            
            for delimiter in delimiters:
                if first_line == delimiter:
                    active_delimiter = delimiter
                    break
            
            # If a delimiter was found, collect multi-line input
            if active_delimiter:
                lines = []
                print(f"(Enter your multi-line text. Type '{active_delimiter}' on a new line when finished)")
                
                # Collect lines until the delimiter is encountered again
                while True:
                    try:
                        line = input()
                        if line.strip() == active_delimiter:
                            break
                        lines.append(line)
                    except EOFError:
                        # Also support Ctrl+D as an alternative way to end input
                        break
                
                return "\n".join(lines)
            else:
                # Single line input
                return first_line
        
        while True:
            # Get user input with history support using our custom function
            try:
                user_input = get_input()
                if user_input:  # Only write non-empty inputs to history
                    readline.add_history(user_input)
                    readline.write_history_file(histfile)
            except KeyboardInterrupt:
                print("\nChat ended by user. Goodbye!")
                break
                
            # Handle commands
            if user_input == '/quit':
                print("\nChat ended. Goodbye!")
                break
            elif user_input == '/new':
                client.new_conversation()
                print("\nStarting new conversation...")
                continue
            elif not user_input:
                continue
                
            # Send message and get response
            client.send_message(user_input)
            
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nChat ended by user. Goodbye!")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()