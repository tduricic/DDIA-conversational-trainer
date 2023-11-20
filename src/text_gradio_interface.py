import gradio as gr
import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the API key and base URL for OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = 'https://api.openai.com/v1'

headers = {
    'Authorization': f'Bearer {OPENAI_API_KEY}',
    'OpenAI-Beta': 'assistants=v1'  # Include the required beta header
}

def create_assistant():
    """Create an assistant in OpenAI."""

    yt_transcript_path = '../data/coffee-machine/yt_instruction_transcript.txt'

    # Open the file and read its contents
    with open(yt_transcript_path, 'r', encoding='utf-8') as file:
        yt_transcript = file.read()

    assistant_instructions = f"""
    You are an AI assistant designed to help users operate a coffee machine and prepare coffee step by step. Your knowledge is based exclusively on a provided manual, 
    which is a transcript from a YouTube video on coffee preparation. This manual includes timestamps for each step in the process. Present each step of the coffee-making process one at a time. 
    After presenting a step, check with the user if they have completed it before moving on to the next one. Your guidance should be clear, concise, and friendly, ensuring a smooth experience for the user.

    Your primary tasks are:
    - To guide users through the process of making coffee, using the steps outlined in the manual.
    - To respond to user queries by referencing specific parts of the manual, using the timestamps to provide context. Important: always add appropriate timestamps to your responses.
    - To assist in troubleshooting common issues during coffee preparation, as detailed in the manual.
    - You should also be able to provide answers in German if the user requests.

    Remember, you do not have external knowledge or access to the internet. All of your responses should be based on the content of the provided manual. 
    You should introduce yourself in the first interaction, explaining your role and how you can assist.
    In subsequent interactions, focus on answering queries and guiding the user through the coffee-making process as per the manual.

    Here is the youtube video transcript you are working with: 
    {yt_transcript}
    """

    response = requests.post(
        f'{BASE_URL}/assistants',
        headers=headers,
        json={
            'name': 'Coffee Machine Tutor',
            'instructions': assistant_instructions,
            'tools': [],
            'model': 'gpt-4-1106-preview'
        }
    )
    assistant = response.json()
    return assistant

def create_thread():
    """Create a thread in OpenAI."""
    response = requests.post(
        f'{BASE_URL}/threads',
        headers=headers
    )
    thread = response.json()
    return thread

def create_message(thread_id, content):
    """Create a message in a thread."""
    response = requests.post(
        f'{BASE_URL}/threads/{thread_id}/messages',
        headers=headers,
        json={
            'role': 'user',
            'content': content
        }
    )

def create_run(thread_id, assistant_id):
    """Create a run in a thread."""
    response = requests.post(
        f'{BASE_URL}/threads/{thread_id}/runs',
        headers=headers,
        json={
            'assistant_id': assistant_id,
        }
    )
    run = response.json()
    return run

def get_run_status(thread_id, run_id):
    """Get the status of a run."""
    response = requests.get(
        f'{BASE_URL}/threads/{thread_id}/runs/{run_id}',
        headers=headers
    )
    run_status = response.json()
    return run_status

def retrieve_messages(thread_id):
    """Retrieve messages from a thread."""
    response = requests.get(
        f'{BASE_URL}/threads/{thread_id}/messages',
        headers=headers
    )
    messages = response.json()
    return messages

def main():
    try:
        assistant = create_assistant()
        thread = create_thread()

        with gr.Blocks() as demo:
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="User Input")
            clear = gr.Button("Clear")

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history):
                user_message = history[-1][0]
                create_message(thread['id'], user_message)

                run = create_run(thread['id'], assistant['id'])

                run_status = get_run_status(thread['id'], run['id'])
                while run_status['status'] != 'completed':
                    time.sleep(2)
                    run_status = get_run_status(thread['id'], run['id'])
                messages = retrieve_messages(thread['id'])
                last_message = next((msg for msg in messages['data'] if msg['role'] == 'assistant' and msg['run_id'] == run['id']), None)
                    
                if last_message and 'content' in last_message and len(last_message['content']) > 0:
                    assistant_response = last_message['content'][0].get('text', {}).get('value', '')
                    bot_message = assistant_response
                else:
                    bot_message = "No response from the assistant."
                history[-1][1] = ""
                for character in bot_message:
                    history[-1][1] += character
                    time.sleep(0.01)
                    yield history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(lambda: None, None, chatbot, queue=False)
            
        demo.queue()
        demo.launch()
    except Exception as error:
        print("Error encountered:", error)

if __name__ == "__main__":
    main()