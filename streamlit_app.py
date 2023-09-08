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
HUB_MAP_PROMPT_REPO = ""
HUB_REDUCE_PROMPT_REPO = ""


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
    map_prompt_template,
    reduce_prompt_template,
    *,
    temperature=0.7,
    num_insights=5,
):
    docs = _get_docs(url)
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
    chain = _construct_chain(llm, map_prompt_template, reduce_prompt_template)
    st.info(chain.run(input_documents=docs, num_insights=num_insights))


st.title('🦜🔗 YouTube Insights')

openai_api_key = st.sidebar.text_input('OpenAI API Key')
lc_hub_api_key = st.sidebar.text_input('LangChainHub API Key')
temperature = st.sidebar.number_input("Model temperature", value=0.7)
num_insights = st.sidebar.number_input("Number of insights", value=5)
with st.form('my_form'):
  url = st.text_area('Enter a YouTube URL:', 'https://youtu.be/ESQkoA8Wx1U')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
      st.warning('Please enter your OpenAI API key!', icon='⚠')
  if not lc_hub_api_key:
      st.warning('Please enter your LangChainHub API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-') and lc_hub_api_key:
      map_prompt_template = hub.pull(
          HUB_MAP_PROMPT_REPO, api_url=HUB_API_URL, api_key=lc_hub_api_key
      )
      st.info(f"Using map prompt: {map_prompt_template.template}")
      reduce_prompt_template = hub.pull(
          HUB_REDUCE_PROMPT_REPO, api_url=HUB_API_URL, api_key=lc_hub_api_key
      )
      st.info(f"Using reduce prompt: {reduce_prompt_template.template}")
      _generate_insights(
          url,
          openai_api_key,
          map_prompt_template,
          reduce_prompt_template,
          temperature=temperature,
          num_insights=num_insights,
      )
