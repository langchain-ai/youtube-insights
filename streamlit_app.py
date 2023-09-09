import os

from langchain import hub
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


HUB_API_URL = "https://api.hub.langchain.com"
HUB_MAP_PROMPT_REPO = "brie/youtube-insights-map"
HUB_REDUCE_PROMPT_REPO = "brie/youtube-insights-reduce"


def _get_docs(url):
    docs = YoutubeLoader.from_youtube_url(url).load()
    return RecursiveCharacterTextSplitter().split_documents(docs)


def _construct_chain(llm, map_prompt_template, reduce_prompt_template):
    document_prompt = PromptTemplate.from_template("{page_content}")
    map_chain = LLMChain(llm=llm, prompt=map_prompt_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt_template)
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
    map_prompt_template,
    reduce_prompt_template,
    *,
    temperature=0.7,
    num_insights=5,
):
    docs = _get_docs(url)
    llm = ChatOpenAI(temperature=temperature)
    chain = _construct_chain(llm, map_prompt_template, reduce_prompt_template)
    st.info(chain.run(input_documents=docs, num_insights=num_insights))


st.title('🦜🔗 YouTube Insights')

temperature = st.sidebar.number_input("Model temperature", value=0.7)
num_insights = st.sidebar.number_input("Number of insights", value=5)
with st.form('my_form'):
  os.environ["LANGCHAIN_API_KEY"]
  if os.environ["LANGCHAIN_TRACING_V2"] != "true":
      raise ValueError
  url = st.text_area('Enter a YouTube URL:', 'https://youtu.be/ESQkoA8Wx1U')
  submitted = st.form_submit_button('Submit')
  if submitted:
      map_prompt_template = hub.pull(
          HUB_MAP_PROMPT_REPO, api_url=HUB_API_URL
      )
      st.info(f"Using map prompt:\n\n{map_prompt_template.template}")
      reduce_prompt_template = hub.pull(
          HUB_REDUCE_PROMPT_REPO, api_url=HUB_API_URL
      )
      st.info(f"Using reduce prompt:\n\n{reduce_prompt_template.template}")
      _generate_insights(
          url,
          map_prompt_template,
          reduce_prompt_template,
          temperature=temperature,
          num_insights=num_insights,
      )
