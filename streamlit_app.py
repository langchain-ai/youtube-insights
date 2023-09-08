from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


DEFAULT_MAP_PROMPT = """Summarize the key insights in the following discussion:

{context}"""

DEFAULT_REDUCE_PROMPT = """You are given summaries of a long discussion. \
Extract from the summaries a numbered list of the top {num_insights} most important \
insights. Each insight should be a single complete sentence:

{context}"""


def _get_docs(url):
    docs = YoutubeLoader.from_youtube_url(url).load()
    return RecursiveCharacterTextSplitter().split_documents(docs)


def _construct_chain(llm, map_prompt_template, reduce_prompt_template):
    document_prompt = PromptTemplate.from_template("{page_content}")
    map_prompt = PromptTemplate.from_template(map_prompt_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_prompt = PromptTemplate.from_template(reduce_prompt_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_prompt=document_prompt,
        document_variable_name="context"
    )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
    )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
    )


def _generate_insights(
    url,
    openai_api_key,
    *,
    temperature=0.7,
    num_insights=5,
    map_prompt_template=DEFAULT_MAP_PROMPT,
    reduce_prompt_template=DEFAULT_REDUCE_PROMPT
):
    docs = _get_docs(url)
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
    chain = _construct_chain(llm, map_prompt_template, reduce_prompt_template)
    st.info(chain.run(input_documents=docs, num_insights=num_insights))

st.title('🦜🔗 YouTube Insights')

openai_api_key = st.sidebar.text_input('OpenAI API Key')
temperature = st.sidebar.number_input("Model temperature", value=0.7)
num_insights = st.sidebar.number_input("Number of insights", value=5)
map_prompt_template = st.sidebar.text_input(
    'Per-chunk prompt',
    value=DEFAULT_MAP_PROMPT
)
reduce_prompt_template = st.sidebar.text_input(
    'Combine prompt',
    value=DEFAULT_REDUCE_PROMPT
)

with st.form('my_form'):
  url = st.text_area('Enter a YouTube URL:', 'https://youtu.be/ESQkoA8Wx1U')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
      st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
      _generate_insights(
          url,
          openai_api_key,
          temperature=temperature,
          num_insights=num_insights,
          map_prompt_template=map_prompt_template,
          reduce_prompt_template=reduce_prompt_template
      )
