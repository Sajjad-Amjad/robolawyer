import os
import streamlit as st
import pinecone
import tempfile

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from streamlit_chat import message

# ----- SESSION STATE MANAGEMENT -----

class SessionState:
    """A class to manage the session state."""
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def get_state():
    """Get or initialize the state."""
    if 'state' not in st.session_state:
        st.session_state['state'] = SessionState(
            index_name="", 
            pinecone_token="", 
            env_name="", 
            system_prompt="", 
            openai_api="",  
            urls=[],  
            num_url_fields=3,  
            uploaded_files_paths=[],  # Store paths of uploaded PDFs
            page="Input Details"
        )
    return st.session_state['state']

# ----- PINECONE UTILITIES -----

def initialize_pinecone(api_key, environment):
    """Initialize Pinecone service."""
    pinecone.init(api_key=api_key, environment=environment)

def create_pinecone_index_for_urls(index_name, openai_api_key, urls):
    """Create a Pinecone index for given URLs."""
    from langchain.document_loaders import SeleniumURLLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings

    # Check if the index already exists
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()

    # Define text splitter based on chunk size
    try:
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=10)
    except:
        text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=10)

    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    # Store the docsearch object in session state
    get_state().docsearch = docsearch
    return docsearch

def create_pinecone_index_for_pdfs(index_name, pdf_dir_path, openai_api_key):
    """Create a Pinecone index for given PDFs."""
    from langchain.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Pinecone
    from langchain.text_splitter import CharacterTextSplitter

    # Check if the index already exists
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

    loader = DirectoryLoader(
        pdf_dir_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    data = loader.load()

    # Define text splitter based on chunk size
    try:
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    except:
        text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    # Store the docsearch object in session state
    get_state().docsearch = docsearch
    return docsearch

# ----- OTHER UTILITIES -----

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory and return its path."""
    temp_dir = tempfile.mkdtemp()  # create a temporary directory
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getvalue())
    return temp_dir

def find_match(query):
    """Find a match for the query using the stored docsearch."""
    docsearch = get_state().docsearch  # Access docsearch stored in session state
    docs = docsearch.similarity_search(query)
    return docs[0].page_content

def get_conversation_string():
    """Generate a conversation string from session state requests and responses."""
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string

# ----- PAGES -----

def input_details_page(state):
    """Render the input details page."""
    st.title("Input Details")

    # Collect various inputs
    state.index_name = st.text_input("Enter Index Name:", value=state.index_name)
    state.pinecone_token = st.text_input("Enter Pinecone Token:", value=state.pinecone_token, type="password")
    state.env_name = st.text_input("Enter Env Name:", value=state.env_name)
    state.system_prompt = st.text_input("Enter System Prompt:", value=state.system_prompt)
    state.openai_api = st.text_input("Enter OpenAI API Key:", value=state.openai_api, type="password")

    # Proceed to next page
    if st.button("Next"):
        try:
            initialize_pinecone(api_key=state.pinecone_token, environment=state.env_name)
            state.page = "Enter URLs or Upload PDFs"
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error initializing Pinecone: {e}")

def upload_urls_page(state):
    """Render the URLs input page or PDF upload page based on user choice."""
    st.title("Enter URLs or Upload PDFs")

    option = st.radio("Choose an option:", ["Enter URLs", "Upload PDFs"])

    if option == "Enter URLs":
        # Collect URLs and dynamically create URL input fields
        all_urls = []
        for i in range(state.num_url_fields):
            url = st.text_input(f"Enter URL {i+1}", key=f"url_{i}")
            all_urls.append(url)

        state.urls = all_urls

        # Add more URL input fields
        if st.button("Add Another URL"):
            state.num_url_fields += 1

        if st.button("Proceed with URLs"):
            try:
                create_pinecone_index_for_urls(index_name=state.index_name, urls=state.urls, openai_api_key=state.openai_api)
                state.page = "Run Query"
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing URLs: {e}")

    elif option == "Upload PDFs":
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            pdf_dir_path = save_uploaded_files(uploaded_files)
            st.write("Files uploaded and saved to:", pdf_dir_path)
            try:
                create_pinecone_index_for_pdfs(index_name=state.index_name, pdf_dir_path=pdf_dir_path, openai_api_key=state.openai_api)
                state.page = "Run Query"
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

def run_query_page(state):
    """Render the run query page."""
    st.title("Run Query")

    # Initialize conversation if it doesn't exist
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    openai_api_key = state.openai_api
    system_msg_template = SystemMessagePromptTemplate.from_template(template=state.system_prompt)
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=2, return_messages=True)

    # Create ChatOpenAI object
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key
    )

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("thinking..."):
                conversation_string = get_conversation_string()
                context = find_match(conversation_string)
                # st.write(context)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')


# ----- MAIN APP -----

def main():
    """Main function to control page flow."""
    state = get_state()

    # Define the routing for pages
    pages = {
        "Input Details": input_details_page,
        "Enter URLs or Upload PDFs": upload_urls_page,
        "Run Query": run_query_page
    }

    # Display the appropriate page based on state.page
    pages[state.page](state)

if __name__ == "__main__":
    main()
