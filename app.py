import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Callable, Generator, Optional, List, Dict
import requests
import json
from consts import AUTO_SEARCH_KEYWORD, SEARCH_TOOL_INSTRUCTION, RELATED_QUESTIONS_TEMPLATE_SEARCH, SEARCH_TOOL_INSTRUCTION, RAG_TEMPLATE, GOOGLE_SEARCH_ENDPOINT, DEFAULT_SEARCH_ENGINE_TIMEOUT, RELATED_QUESTIONS_TEMPLATE_NO_SEARCH
import re
import asyncio
import random

import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


from visual_env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

logging.basicConfig(level=logging.INFO)
GOOGLE_API_KEY = st.secrets["google_api_key"]
GOOGLE_CX = st.secrets["google_cx"]
BACKUP_KEYS = [st.secrets["backup_key_1"], st.secrets["backup_key_2"], st.secrets["backup_key_3"], st.secrets["backup_key_4"], st.secrets["backup_key_5"]]

CONFIG_PATH = os.path.join(current_dir, "config.yaml")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
]

def load_config():
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


config = load_config()
prod_mode = config.get('prod_mode', False)
additional_env_vars = config.get('additional_env_vars', None)

@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator:
    """
    context manager to catch stdout and send it to an output streamlit element
    Args:
        output_func (function to write terminal output in
    Yields:
        Generator:
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write  # type: ignore
        yield

async def run_samba_api_inference(query, system_prompt = None, ignore_context=False, max_tokens_to_generate=None, num_seconds_to_sleep=1, over_ride_key=None):
    # First construct messages
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    if not ignore_context:
        for ques, ans in zip(
            st.session_state.chat_history[::3],
            st.session_state.chat_history[1::3],
        ):
            messages.append({"role": "user", "content": ques})
            messages.append({"role": "assistant", "content": ans})
    messages.append({"role": "user", "content": query})

    # Create payloads
    payload = {
        "messages": messages,
        "model": config.get("model")
    }
    if max_tokens_to_generate is not None:
        payload["max_tokens"] = max_tokens_to_generate

    if over_ride_key is None:
        api_key = st.session_state.SAMBANOVA_API_KEY
    else:
        api_key = over_ride_key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        post_response = await asyncio.get_event_loop().run_in_executor(None, lambda: requests.post(config.get("url"), json=payload, headers=headers, stream=True))
        post_response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if post_response.status_code in {401, 503}:
            st.info(f"Invalid Key! Please make sure you have a valid SambaCloud key from https://cloud.sambanova.ai/.")
            return "Invalid Key! Please make sure you have a valid SambaCloud key from https://cloud.sambanova.ai/."
        if post_response.status_code in {429, 504}:
            await asyncio.sleep(num_seconds_to_sleep)
            return await run_samba_api_inference(query, over_ride_key=random.choice(BACKUP_KEYS))  # Retry the request
        else:
            print(f"Request failed with status code: {post_response.status_code}. Error: {e}")
            return "Invalid Key! Please make sure you have a valid SambaCloud key from https://cloud.sambanova.ai/."

    response_data = json.loads(post_response.text)

    return response_data["choices"][0]["message"]["content"]

def extract_query(text):
    # Regular expression to capture the query within the quotes
    match = re.search(r'query="(.*?)"', text)
    
    # If a match is found, return the query, otherwise return None
    if match:
        return match.group(1)
    return None

def extract_text_between_brackets(text):
    # Using regular expressions to find all text between brackets
    matches = re.findall(r'\[(.*?)\]', text)
    return matches

def search_with_google(query: str):
    """
    Search with google and return the contexts.
    """
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": 5,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )

    if not response.ok:
        raise Exception(response.status_code, "Search engine error.")
    json_content = response.json()

    contexts = json_content["items"][:5]

    return contexts

async def get_related_questions(query, contexts = None):
    if contexts:
        related_question_system_prompt = RELATED_QUESTIONS_TEMPLATE_SEARCH.format(
            context="\n\n".join([c["snippet"] for c in contexts])
        )
    else:
        # When no search is performed, use a generic prompt
        related_question_system_prompt = RELATED_QUESTIONS_TEMPLATE_SEARCH

    related_questions_raw = await run_samba_api_inference(query, related_question_system_prompt)

    try:
        return json.loads(related_questions_raw)  
    except:
        try:
            extracted_related_questions = extract_text_between_brackets(related_questions_raw)
            return json.loads(extracted_related_questions)
        except:
            return []

def process_citations(response: str, search_result_contexts: List[Dict]) -> str:
    """
    Process citations in the response and replace them with numbered icons.
    
    Args:
        response (str): The original response with citations.
        search_result_contexts (List[Dict]): The search results with context information.
    
    Returns:
        str: The processed response with numbered icons for citations.
    """
    citations = re.findall(r'\[citation:(\d+)\]', response)
    
    for i, citation in enumerate(citations, 1):
        response = response.replace(f'[citation:{citation}]', f'<sup>[{i}]</sup>')
    
    return response

def generate_citation_links(search_result_contexts: List[Dict]) -> str:
    """
    Generate HTML for citation links.
    
    Args:
        search_result_contexts (List[Dict]): The search results with context information.
    
    Returns:
        str: HTML string with numbered citation links.
    """
    citation_links = []
    for i, context in enumerate(search_result_contexts, 1):
        title = context.get('title', 'No title')
        link = context.get('link', '#')
        citation_links.append(f'<p>[{i}] <a href="{link}" target="_blank">{title}</a></p>')
    
    return ''.join(citation_links)

        
async def run_auto_search_pipe(query):
    full_context_answer = asyncio.create_task(run_samba_api_inference(query))
    related_questions_no_search = asyncio.create_task(get_related_questions(query))

    # First call Llama3.1 8B with special system prompt for auto search
    with st.spinner('Checking if web search is needed...'):
        auto_search_result = await run_samba_api_inference(query, SEARCH_TOOL_INSTRUCTION, True, max_tokens_to_generate=100)

    # If Llama3.1 8B returns a search query then run search pipeline
    if AUTO_SEARCH_KEYWORD in auto_search_result:
        st.session_state.search_performed = True
        # search
        with st.spinner('Searching the internet...'):
            search_result_contexts = search_with_google(extract_query(auto_search_result))

        # RAG response
        with st.spinner('Generating response based on web search...'):
            rag_system_prompt = RAG_TEMPLATE.format(
                context="\n\n".join(
                    [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(search_result_contexts)]
                )
            )

            model_response = asyncio.create_task(run_samba_api_inference(query, rag_system_prompt))
            related_questions = asyncio.create_task(get_related_questions(query, search_result_contexts))
            # Process citations and generate links
            citation_links = generate_citation_links(search_result_contexts)
            
            model_response_complete = await model_response
            processed_response = process_citations(model_response_complete, search_result_contexts)
            related_questions_complete = await related_questions


        return processed_response, citation_links, related_questions_complete
    
    # If Llama3.1 8B returns an answer directly, then please query Llama 405B to get the best possible answer
    else:
        st.session_state.search_performed = False
        result = await full_context_answer
        related_questions = await related_questions_no_search
        return result, "", related_questions


def handle_userinput(user_question: Optional[str]) -> None:
    """
    Handle user input and generate a response, also update chat UI in streamlit app
    Args:
        user_question (str): The user's question or input.
    """
    if user_question:
        # Clear any existing related question buttons
        if 'related_questions' in st.session_state:
            st.session_state.related_questions = []

        async def run_search():
            return await run_auto_search_pipe(user_question)
        
        response, citation_links, related_questions = asyncio.run(run_search())
        if st.session_state.search_performed:
            search_or_not_text = "🔍 Web search was performed for this query."
        else:
            search_or_not_text = "📚 This response was generated from the model's knowledge."

        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append((response, citation_links))
        st.session_state.chat_history.append(search_or_not_text)

        # Store related questions in session state
        st.session_state.related_questions = related_questions

    for ques, ans, search_or_not_text in zip(
        st.session_state.chat_history[::3],
        st.session_state.chat_history[1::3],
        st.session_state.chat_history[2::3],
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')
    
        with st.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            st.markdown(f'{ans[0]}', unsafe_allow_html=True)
            if ans[1]:
                st.markdown("### Sources", unsafe_allow_html=True)
                st.markdown(ans[1], unsafe_allow_html=True)
            st.info(search_or_not_text)
    if len(st.session_state.related_questions) > 0:
        st.markdown("### Related Questions")
        for question in st.session_state.related_questions:
            if st.button(question):
                setChatInputValue(question)

def setChatInputValue(chat_input_value: str) -> None:
    js = f"""
    <script>
        function insertText(dummy_var_to_force_repeat_execution) {{
            var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            nativeInputValueSetter.call(chatInput, "{chat_input_value}");
            var event = new Event('input', {{ bubbles: true}});
            chatInput.dispatchEvent(event);
        }}
        insertText(3);
    </script>
    """
    st.components.v1.html(js)

def main() -> None:
    st.set_page_config(
        page_title='Auto Web Search Demo',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )


    initialize_env_variables(prod_mode, additional_env_vars)

    if 'input_disabled' not in st.session_state:
        if 'SAMBANOVA_API_KEY' in st.session_state:
            st.session_state.input_disabled = False
        else:
            st.session_state.input_disabled = True
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'related_questions' not in st.session_state:
        st.session_state.related_questions = [] 

    st.title(' Auto Web Search')
    st.subheader('Powered by :orange[SambaNova Cloud] and Llama405B')

    with st.sidebar:
        st.title('Get your :orange[SambaNova Cloud] API key [here](https://cloud.sambanova.ai/apis)')

        if not are_credentials_set(additional_env_vars):
            api_key, additional_vars = env_input_fields(additional_env_vars)
            if st.button('Save Credentials'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.session_state.input_disabled = False
                st.success(message)
                st.rerun()

        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials'):
                save_credentials('', {var: '' for var in (additional_env_vars or [])}, prod_mode)
                st.session_state.input_disabled = True
                st.rerun()


        if are_credentials_set(additional_env_vars):
            with st.expander('**Example Queries With Search**', expanded=True):
                if st.button('What is the population of Virginia?'):
                    setChatInputValue(
                        'What is the population of Virginia?'
                    )
                if st.button('SNP 500 stock market moves'):
                    setChatInputValue('SNP 500 stock market moves')
                if st.button('What is the weather in Palo Alto?'):
                    setChatInputValue(
                        'What is the weather in Palo Alto?'
                    )
            with st.expander('**Example Queries No Search**', expanded=True):
                if st.button('write a short poem following a specific pattern: the first letter of every word should spell out the name of a country.'):
                    setChatInputValue(
                        'write a short poem following a specific pattern: the first letter of every word should spell out the name of a country.'
                    )
                if st.button('Write a python program to find the longest root to leaf path in a tree, and some test cases for it.'):
                    setChatInputValue('Write a python program to find the longest root to leaf path in a tree, and some test cases for it.')

            st.markdown('**Reset chat**')
            st.markdown('**Note:** Resetting the chat will clear all interactions history')
            if st.button('Reset conversation'):
                st.session_state.chat_history = []
                st.session_state.sources_history = []
                if 'related_questions' in st.session_state:
                    st.session_state.related_questions = []
                st.toast('Interactions reset. The next response will clear the history on the screen')

        # Add a footer with the GitHub citation
        footer_html = """
        <style>
        .footer {
            position: fixed;
            right: 10px;
            bottom: 10px;
            width: auto;
            background-color: transparent;
            color: grey;
            text-align: right;
            padding: 10px;
            font-size: 16px;
        }
        </style>
        <div class="footer">
            Inspired by: <a href="https://github.com/leptonai/search_with_lepton" target="_blank">search_with_lepton</a>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)

    user_question = st.chat_input('Ask something', disabled=st.session_state.input_disabled, key='TheChatInput')
    handle_userinput(user_question)

    

if __name__ == '__main__':
    main()