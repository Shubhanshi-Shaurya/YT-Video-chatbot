from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st 
from langchain_ollama import ChatOllama
import json
import urllib.request

def get_video_title(video_id):
    try:
      url=f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
      with urllib.request.urlopen(url) as response:
        data=json.loads(response.read().decode())
        return data.get("title","Unknown video")
      
    except Exception:
      return "Youtube Video"

st.set_page_config("YT Chatbot")
st.title("YT Video Chatbot")

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    llm = ChatOllama(model="llama3", temperature=0.2)
    return embeddings,llm

embeddings,llm=load_models()

with st.sidebar:
  st.header("Video Management")
  video_id=st.text_input("Enter the Youtube video ID")
   
  process_button=st.button("Process Video",type="primary")

  if process_button:
     title=get_video_title(video_id)
     thumbnail_url=f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
     st.image(thumbnail_url,caption=title,use_container_width=True)

if video_id and process_button:
  with st.spinner("Processing Video"):
    try:
      fetched_transcipt=YouTubeTranscriptApi().fetch(video_id,languages=['en'])
      transcript_list = fetched_transcipt.to_raw_data()

      transcript = " ".join(chunk["text"] for chunk in transcript_list)
      # print(transcript)

      splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
      chunks=splitter.create_documents([transcript])

      vector_store=FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
      )
      
      # vector_store.index_to_docstore_id
      # vector_store.get_by_ids([''])
      retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

      prompt = ChatPromptTemplate.from_template(
              """
              You are a helpful assistant.
              Use the retrieved context to answer the user's question.
              The context may be in a different language than the question.
              Answer in the same language as the user's question.

              Context:
              {context}

              Question:
              {question}

              Answer:
              """.strip()
          )

      def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

      parallel_chain=RunnableParallel({
      'context':retriever | RunnableLambda(format_docs),
      'question':RunnablePassthrough()
      })

      main_chain= parallel_chain | prompt | llm | StrOutputParser()

      st.session_state.main_chain = main_chain
      st.session_state.messages = []
      st.success("Video processed successfully! Ask your questions below.")

    except TranscriptsDisabled:
      print("No captions available for this video.")
          
    except Exception as e:
      print(f"an unexpected error occured {e}")


st.subheader("Chat with your video")

if "messages" not in st.session_state:
  st.session_state.messages = []


for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])


if "main_chain" in st.session_state:
    if user_query := st.chat_input("Ask a question about your video...."):
        
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching video and thinking...."):
                response = st.session_state.main_chain.invoke(user_query)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please enter a YouTube video ID and click 'Process Video' in the sidebar to begin chatting.")


