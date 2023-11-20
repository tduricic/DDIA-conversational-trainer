# Import necessary modules and packages
import sys
import os
import argparse
import logging
import requests
import json
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

API_KEY = "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"
os.environ["OPENAI_API_KEY"] = API_KEY

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Conversational trainer for a coffee machine.")
    parser.add_argument("--pdf_url", required=False, default="https://decentespresso.com/doc/quickstart/quickstart.pdf", help="URL to the PDF of the instruction manual.")
    parser.add_argument("--yt_video_id", required=False, default="S74uI686Hkg", help="URL to the YouTube video with instructions.")
    args = parser.parse_args()

    parent_directory_path = "../data/coffee-machine/"
    create_parent_directories(parent_directory_path)
    instruction_manual_path = parent_directory_path + "instruction-manual.pdf"
    if not os.path.exists(instruction_manual_path):
        download_pdf(args.pdf_url, instruction_manual_path)

    transcript_path = parent_directory_path + "yt_instruction_transcript.txt"
    if not os.path.exists(transcript_path):
        download_and_store_video_transcript(args.yt_video_id, transcript_path)

    video_filename = "yt_instruction_video.mp4"
    if not os.path.exists(parent_directory_path + video_filename):
        download_and_store_video(args.yt_video_id, parent_directory_path, video_filename)

    PERSIST = False

    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = TextLoader("../data/coffee-machine/yt_instruction_transcript.txt")
        # loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        print(result['answer'])

        chat_history.append((query, result['answer']))
        query = None

def create_parent_directories(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_pdf(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, "wb") as pdf_file:
            pdf_file.write(response.content)
        print("PDF file downloaded successfully.")
    else:
        print("Failed to download PDF file.")

def download_and_store_video_transcript(id, destination):
    transcript = YouTubeTranscriptApi.get_transcript(id)
    with open(destination, "w") as transcript_file:
        json.dump(transcript, transcript_file, indent=4)

def download_and_store_video(id, parent_directory_path, video_filename):
    yt = YouTube("https://www.youtube.com/watch?v=" + id)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(output_path=parent_directory_path, filename=video_filename)

if __name__ == "__main__":
    main()