import requests
import json
import argparse
import os
import readline
from dotenv import load_dotenv

# Fixed endpoint path
API_ENDPOINT = "/v1/chat/completions"

def get_api_base():
    # Load environment variables from .env file
    load_dotenv()
    
    # Command line arguments have highest priority
    parser = argparse.ArgumentParser(description='Interactive chat client')
    parser.add_argument('--api-base', help='API base URL')
    args = parser.parse_args()
    
    # Priority order: command line > environment variable
    if args.api_base:
        return args.api_base.rstrip('/')
    elif 'CHAT_API_BASE' in os.environ:
        return os.environ['CHAT_API_BASE'].rstrip('/')
    else:
        raise ValueError("API base URL not found. Please set it via --api-base argument or CHAT_API_BASE environment variable in .env file")

def chat_with_model(messages, api_base):
    response = requests.post(
        f"{api_base}{API_ENDPOINT}",
        json={
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.1
        },
        stream=True
    )

    print("\nAssistant:", end=' ', flush=True)
    assistant_message = ""
    for line in response.iter_lines():
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
                            print(content, end='', flush=True)
                            assistant_message += content
                except json.JSONDecodeError:
                    continue
    print()  # Add a newline at the end
    return assistant_message

def get_initial_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant"}
    ]

def main():
    try:
        api_base = get_api_base()
        messages = get_initial_messages()
        
        # Configure readline for history
        histfile = ".chat_history"  # Store in current working directory
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        
        print(f"Using API base URL: {api_base}")
        print("\nChat started. Available commands:")
        print("  /quit - Exit the chat")
        print("  /new  - Start a new conversation")
        print("  Use ↑/↓ keys to navigate through history")
        
        while True:
            # Get user input with history support
            user_input = input("\nYou: ").strip()
            readline.write_history_file(histfile)
            
            # Handle commands
            if user_input == '/quit':
                print("\nChat ended. Goodbye!")
                break
            elif user_input == '/new':
                messages = get_initial_messages()
                print("\nStarting new conversation...")
                continue
                
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Get model response
            assistant_message = chat_with_model(messages, api_base)
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": assistant_message})
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
