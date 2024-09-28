import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.chains import create_history_aware_retriever , create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import io
from fpdf import FPDF
import requests
import json
import validators
import time
import re

import validators.url

# Page configuration
st.set_page_config(page_title="Triumo Chatbot: App With Multiple Functionalities")

api_key = "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"

# Option Selection
st.title("Triumo Chatbot: App With Multiple Functionalities")
st.subheader("Choose an option to proceed:")

# Options
option = st.selectbox("Choose an action:", ["Select", "Chat With LLM" , "Summarize a URL" , "Summarise the PDF And Ask Questions" , "Code-Problem Solver" , "Web Scrape and Search the text" , "ML Evaluation and PDF Generation"])

# Initialize the Groq model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key)

# Initialise the type of embeddings
embeddings = OllamaEmbeddings()
retriever_tool = None

# Initializing the Wikipedia and Arxiv tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchResults(name="Search")

retriever_tool = None  # Initialize as None in case no file is uploaded
embeddings = OllamaEmbeddings()

# ------------------------- Normal Chatbot Functonality ------------------------------ #
if option == "Chat With LLM":
    # Initialize session state for messages (conversation history)
  if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I assist you?"}
    ]

  # Display previous messages in the chat
  for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

  # Handling the user prompt
  if prompt := st.chat_input(placeholder="Ask anything..."):
    # Add the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the model and tools
    llm = ChatGroq(api_key=api_key, model="gemma2-9b-it", streaming=True)

    # Include available tools in the agent
    tools = [arxiv, wiki, search]

    # Create the search agent
    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION , max_iterations=15   # Increase this value as per your needs
    )

    # Construct the conversation history as a single string
    conversation_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
    )

    # Run the agent with the full conversation history
    response = search_agent.run(conversation_history)

    # Add assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# ------------------- Summarize a URL -------------------
elif option == "Summarize a URL":
    # Summarize a URL
    st.subheader("Summarize content from a YouTube video or a Website URL")
    url_input = st.text_input("Enter a valid URL:", label_visibility="collapsed")
    
    # Define the prompts for summarization
    chunks_prompt = """
    Please summarize the below speech:
    Speech:{text}
    Summary:
    """
    final_prompt = """
    Provide the final summary of the entire speech with these important points.
    Add a motivational Title, start the summary with an introduction, and provide the summary in numbered points.
    Speech:{text}
    """
    map_prompt_template = PromptTemplate(input_variables=["text"], template=chunks_prompt)
    final_prompt_template = PromptTemplate(input_variables=["text"], template=final_prompt)

    # Function to handle transcript unavailability
    def get_transcript_in_available_language(url):
        try:
            return YoutubeLoader.from_youtube_url(url, add_video_info=True).load()
        except Exception as e:
            if "No transcripts were found" in str(e):
                st.warning("No transcript in the requested language. Trying alternative language options...")
                return YoutubeLoader.from_youtube_url(url, add_video_info=True, language="hi").load()  # Hindi as fallback
            else:
                raise e

    # Dynamic chunking based on document length
    def dynamic_chunk_size(docs, max_size=5000, min_size=1000, overlap=1000):
        chunk_size = max_size
        num_chunks = len(RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_documents(docs))

        while num_chunks > 20 and chunk_size > min_size:
            chunk_size -= 500
            num_chunks = len(RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_documents(docs))

        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_documents(docs)

    if url_input and st.button("Summarize the URL"):
        if not validators.url(url_input):
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Processing URL..."):
                try:
                    if "youtube.com" in url_input:
                        docs = get_transcript_in_available_language(url_input)
                    else:
                        loader = UnstructuredURLLoader(urls=[url_input], ssl_verify=True, headers={"User-Agent": "Mozilla/5.0"})
                        docs = loader.load()

                    final_documents = dynamic_chunk_size(docs)
                    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=final_prompt_template)
                    output_summary = chain.run(final_documents)

                    st.success(output_summary)

                except Exception as e:
                    st.exception(f"Error: {e}")

# ------------------- Summarize a PDF and Highilight the Inportant Section -------------------#
# Summarize PDF option
elif option == "Summarise the PDF And Ask Questions":
    session_id = st.text_input("Session Id:", value="default Session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # taking the PDF from the user to summarise it
    uploaded_files = st.file_uploader("Choose a PDF file:", type="pdf", accept_multiple_files=True)
    documents = []
    
    if uploaded_files:
        with st.spinner("Processing the PDF files, Please wait..."):
            for uploaded_file in uploaded_files:
                temppdf = f"./{uploaded_file.name}"  # Unique name for each file
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(temppdf)
                docs = loader.load_and_split()
                documents.extend(docs)  # Use extend to flatten the list of docs

        # Splitting the documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        final_documents = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=final_documents, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Creating a prompt template for summarization
        chunks_prompt = """
        Write a short and concise summary of the following speech,
        Speech: {text}
        Summary:
        """
        map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

        final_prompt = """
        Provide the final summary of the entire speech with these important points.
        Add a motivational Title, Start the precise summary with an introduction and provide the summary in numbered points for the speech,
        Speech: {text}
        """
        final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

        # Load summarization chain
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=True
        )

        # Run the summarization chain
        output = summary_chain.run(final_documents)
        st.success(output)

        # Define the retriever for the RAG chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer Question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer say that you don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to get the session history from session state
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Taking user input for question-answering
        user_input = st.text_input("Your Question:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            # Display conversation history and answer
            # st.write(f"Session ID: {session_id}")
            st.write(response['answer'])
            # st.write("Chat History:", session_history.messages)

# ------------------- Code-Problem Solver -------------------#
elif option == "Code-Problem Solver":
    url = "http://localhost:11434/api/generate"

    headers = {
        'Content-Type': 'application/json'
    }

    history = []

    def generate_response(prompt):
        history.append(prompt)
        final_prompt = "\n".join(history)

        data = {
            "model": "codeguru",
            "prompt": final_prompt,
            "stream": False
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            response_text = response.text
            data = json.loads(response_text)
            actual_data = data['response']
            return actual_data
        else:
            return f"Error: {response.text}"

    # Streamlit UI
    st.title("Chatbot Web App")

    prompt = st.text_area("Enter your prompt", height=150)

    if st.button("Generate Response"):
        if prompt:
            response = generate_response(prompt)
            st.text_area("Response", value=response, height=200)
        else:
            st.write("Please enter a prompt")

# ------------------- Web Scrape the Website and Search the text -------------------#
elif option == "Web Scrape and Search the text":
    st.subheader("Web Scape a Website and Search for the text")

    url = st.text_input("Enter the URL you want to scrape: ")
    search_text = st.text_input("Enter the text or pattern to search for: ")

    # Function to scarpe the website
    def web_scrape_and_search(url , search_text):
        try:
            if not validators.url(url):
                return "Invalid URL. Please enter a valid webiste URL."
            
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content , "lxml")
                website_text = soup.get_text()

                ocurrences = re.findall(search_text , website_text , re.IGNORECASE)

                if ocurrences:
                    result = f"Found {len(ocurrences)} occurences of '{search_text}"
                    return result
                else:
                    return f"Failed to retrieve the website.Status code: {response.status_code}"
        except Exception as e:
            return f"An error occured: {e}"

    if st.button("Search"):
        if url and search_text:
            result = web_scrape_and_search(url , search_text)
            st.success(result)
        else:
            st.error("Please enter both the URL and the text to search for.")   

# ------------------- ML Evaluation and PDF Generation -------------------#    
elif option == "ML Evaluation and PDF Generation":
    def chatbot_response(user_input):
        responses = {
            "start": "Welcome! I can help you evaluate machine learning models. Please upload a dataset to get started.",
            "upload": "Please upload your dataset (CSV format).",
            "task": "What task would you like to perform?",
            "select_features": "Please select your features and target columns.",
            "evaluate": "I am evaluating the results now... Here are the results.",
            "error": "Oops! Something went wrong. Please make sure you upload a valid dataset."
        }
        return responses.get(user_input.lower(), "I didn't understand that. Could you try again?")

    # Helper function to generate PDF
    def generate_pdf(report_content, plot_files, dataframe, selected_features, target_feature , task , model_name , model_params , metrics):
        pdf = FPDF()
        # pdf.add_page()

        # # Add textual content (general report content)
        # pdf.set_font("Arial", size=12)
        # for line in report_content:
        #     pdf.multi_cell(0, 10, line)

        pdf.add_page()
        pdf.set_font("Arial" , style="UBI" , size=15)
        pdf.cell(200 , 10 , txt="Research Paper Published" , ln=True , align="C")
        pdf.ln(15)

        # Add all feature names without cells
        pdf.set_font("Arial", style='UB', size=12)
        pdf.cell(200, 10, txt="All Features In the Dataset", ln=True, align="C")
        for feature in dataframe.columns:
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 7, f"- {feature}")  # Use multi_cell for line breaks

         # Add task and model name to the PDF
        pdf.set_font("Arial", style='UB', size=12)
        pdf.cell(200, 10, txt="Technologies and Models Used", ln=True, align="C")
        pdf.set_font("Arial", size=10)
        pdf.ln(10)  # Add some spacing
        pdf.cell(200, 10, txt=f"Task Performed: {task}", ln=True)
        pdf.cell(200, 10, txt=f"Model Used: {model_name}", ln=True)
    

        # New page for features used in analysis
        pdf.add_page()
        pdf.set_font("Arial", style='UB', size=12)  # Set font to bold and underline
        pdf.cell(200, 10, txt="Features Used for Data Analysis", ln=True, align="C")
        pdf.set_font("Arial" , size=10)
        pdf.cell(200, 10, txt="Top 5 Data Of the Selected Features will be Displayed", ln=True, align="C")

        # Create a table for selected features and target feature
        pdf.set_font("Arial", size=10)

        # Calculate maximum widths for each column based on headers and data
        column_widths = []
        headers = selected_features + [target_feature]

        for header in headers:
            max_width = pdf.get_string_width(str(header))
            column_widths.append(max_width + 10)  # Add some padding

        # Add column headers
        for header, width in zip(headers, column_widths):
            pdf.cell(width, 10, header, 1, 0, 'C')  # Adjust column width dynamically
        pdf.ln()

        # Add first 5 rows of data
        for i in range(min(5, len(dataframe))):  # Limit to 5 rows
            for col, width in zip(headers, column_widths):
                pdf.cell(width, 10, str(dataframe.iloc[i][col]), 1, 0, 'C')  # Adjust column width
            pdf.ln()  # Move to next line after each row

        # For feature Statistics Summary
        pdf.set_font("Arial" , style='UB' , size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt="Dataset Summary Statistics", ln=True, align="C")
        pdf.set_font("Arial", size=10)

        # Get summary statistics
        summary_stats = dataframe.describe().transpose()  # Transpose to have features as rows
        summary_headers = summary_stats.columns.tolist()
        summary_data = summary_stats.values.tolist()

        # Calculate maximum widths for summary statistics columns
        summary_column_widths = [max(pdf.get_string_width(str(item)) for item in [header] + [row[i] for row in summary_data]) + 10
                                for i, header in enumerate(summary_headers)]

        # Add summary headers to the PDF
        for header, width in zip(summary_headers, summary_column_widths):
          pdf.cell(width, 10, header, 1, 0, 'C')  # Adjust column width dynamically
        pdf.ln()

        # Add summary data to the PDF
        for row in summary_data:
          for value, width in zip(row, summary_column_widths):
             pdf.cell(width, 10, str(value), 1, 0, 'C')  # Adjust column width
          pdf.ln()  # Move to next line after each row

        # Add model parameters and hyperparameters
        pdf.add_page()
        pdf.set_font("Arial", style='UB', size=12)
        pdf.cell(200, 10, txt="Model Parameters and Hyperparameters", ln=True, align="C")
        pdf.set_font("Arial", size=10)
        for param, value in model_params.items():
            pdf.cell(200, 10, txt=f"{param}: {value}", ln=True)

        pdf.set_font("Arial", style='UB', size=12)
        pdf.cell(200, 10, txt="Model Performance Metrics", ln=True, align="C")
        pdf.set_font("Arial", size=10)
        for metric, value in metrics.items():
            pdf.cell(200, 10, txt=f"{metric}: {value}", ln=True)    

        # Add plots to the PDF
        for plot_file in plot_files:
            pdf.add_page()  # New page for each plot
            pdf.image(plot_file, x=10, y=10, w=180)  # Adjust the plot size as needed

        # Output PDF as string and convert to BytesIO
        pdf_output = pdf.output(dest='S').encode('latin1')  # Encode in latin1 for compatibility with PDF
        return io.BytesIO(pdf_output)  # Convert string to BytesIO object

    # Streamlit Chatbot Interface
    st.title("ML Model Evaluation Chatbot")
    st.write("Just type 'upload' in the below box and then enter to upload CSV files for Data Analysis")
    user_input = st.text_input("You: ", "")

    # Initialize report content list and list for plot files
    report_content = []
    plot_files = []

    # Chatbot flow control
    if user_input.lower() == "start":
        st.write(chatbot_response("start"))
    elif user_input.lower() == "upload":
        st.write(chatbot_response("upload"))

        # File Upload option
        uploaded_file = st.file_uploader("Upload your dataset file:", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:", data.head())
            report_content.append("Dataset Preview:\n")
            report_content.append(data.head().to_string())
            report_content.append("\n")

            # Task Selection
            user_input_task = st.selectbox("What task would you like to perform?", ["Classification", "Regression"])

            # Chatbot asks for feature and target selection
            if user_input_task:
                st.write(chatbot_response("select_features"))

                data = data.dropna()  # Removing null values
                # Select target and features
                target_column = st.selectbox("Select the Target Column:", data.columns)
                feature_column = st.multiselect("Select the feature columns:", data.columns.difference([target_column]))

                if len(feature_column) > 0:
                    # Split into target and features
                    X = data[feature_column]
                    y = data[target_column]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train and evaluate the model
                    st.write(chatbot_response("evaluate"))
                    model_params = {}
                    metrics = {}
                    if user_input_task == "Classification":
                        models = {
                            "Logistic Regression": LogisticRegression(),
                        }
                        results = {}
                        for model_name, model in models.items():
                            model_name_used = model_name
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_pred, y_test)
                            results[model_name] = accuracy
                            model_params = model.get_params()
                            metrics['Accuracy'] = accuracy
                            st.write(f"{model_name} Accuracy: {accuracy:.4f}")
                            report_content.append(f"{model_name} Accuracy: {accuracy:.4f}\n")

                        # Plot model accuracies
                        fig, ax = plt.subplots()
                        ax.barh(list(results.keys()), list(results.values()))
                        st.pyplot(fig)

                        # Save the plot to a temporary file and store the file path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                            plt.savefig(tmpfile.name)
                            plot_files.append(tmpfile.name)

                        # Draw plots between target and feature columns (For Classification)
                        for feature in feature_column:
                            fig, ax = plt.subplots()
                            try:
                                sns.barplot(x=data[feature], y=y)
                                plt.title(f'Bar Plot - {feature} vs {target_column}')
                                st.pyplot(fig)

                                # Save the feature-target plot to temporary files
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                                    plt.savefig(tmpfile.name)
                                    plot_files.append(tmpfile.name)
                            except Exception as e:
                                st.error(f"Error generating bar plot for {feature}: {e}")

                    elif user_input_task == "Regression":
                        models = {
                            "Linear Regression": LinearRegression(),
                        }
                        results = {}
                        for model_name, model in models.items():
                            model_name_used = model_name
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            results[model_name] = mse
                            model_params = model.get_params()
                            metrics['Mean Squared Error'] = mse
                            st.write(f"{model_name} Mean Squared Error: {mse:.4f}")
                            report_content.append(f"{model_name} Mean Squared Error: {mse:.4f}\n")

                        # Plot model errors
                        fig, ax = plt.subplots()
                        ax.barh(list(results.keys()), list(results.values()))
                        st.pyplot(fig)

                        # Save the plot to a temporary file and store the file path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                            plt.savefig(tmpfile.name)
                            plot_files.append(tmpfile.name)

                        # Draw plots between target and feature columns (For Regression)
                        for feature in feature_column:
                            fig, ax = plt.subplots()
                            try:
                                plt.scatter(data[feature], y, alpha=0.5)
                                plt.xlabel(feature)
                                plt.ylabel(target_column)
                                plt.title(f'Scatter Plot - {feature} vs {target_column}')
                                st.pyplot(fig)

                                # Save the feature-target plot to temporary files
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                                    plt.savefig(tmpfile.name)
                                    plot_files.append(tmpfile.name)
                            except Exception as e:
                                st.error(f"Error generating scatter plot for {feature}: {e}")

                    # Convert report content and plot images to downloadable PDF
                    pdf_output = generate_pdf(report_content, plot_files, data, feature_column, target_column , user_input_task , model_name_used , model_params , metrics)

                    # Allow users to download the report as a PDF with plots
                    st.download_button(
                        label="Download Full Report with Plots as PDF",
                        data=pdf_output,
                        file_name="ml_evaluation_report.pdf",
                        mime="application/pdf"
                    )

                else:
                    st.write("Please select at least one feature column.")
            else:
                st.write(chatbot_response(user_input.lower()))