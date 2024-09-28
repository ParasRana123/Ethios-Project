# TRUMIO Chat Bot     
The Trumio Chatbot is an application that offers users with multiple functionalities as mentioned below. It is an application that reduces the need of the users to switch to different platforms or documents in order to perform various tasks. Also it ends the need of Manual Intervention to train a model and generate PDF containing various useful Insights.
        
## FEATURES OF THE APP:       

+ Chat with LLM
+ Summarisation Of URL and PDF
+ ML Model Selection and PDF Generation
+ Code Problem Solver
+ Web Scraping and Searching for a specific text or pattern    

## How To run this project

1. Clone or Download this Repository to your local machine.
2. Then go to your Command prompt in VS Code and create a virtual environment venv with a python version == 3.12.0 using the command **conda create -p python == 3.12 -y**
3. Install all the libraries mentioned in the requirements.txt file with the command **pip install -r requirements.txt**
4. Also install llama2 model on your local machine by writing the command **ollama run llama2**
5. For using the code-problem solver funnctionality , go to your command promple and go to the loaction of your modelfile using the 'cd' command and then write **ollama create codeguru -f modelfile** and then **ollama run codeguru** this will run the model in background
6. Open your terminal/command prompt from your project directory and run the file main.py by executing the command **streamlit run main.py**
7. Go to your browser and type **http://192.168.174.134:8501**
8. Hurray! That's it.
