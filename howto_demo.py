import json
import logging
import os
import sys
from typing import List

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pydantic import BaseModel, Field
import streamlit as st

# ===== Query Engine Setup ===== #

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

STORAGE = "./storage/howto-212k-bge-small-en-384"
EMBED = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384
RESPONSE_MODE = "tree_summarize"
COLLECTION = "howto212k"


class HowToStep(BaseModel):
    """How to step data model"""

    description: str = Field(
        description=
        "description of specific step to take in series of steps to solve problem / accomplish task"
    )
    url: str = Field(description="link to the source youtube video")
    scene: int = Field(
        description=
        "relevant video scene number in source video, only provide an integer number"
    )
    timestamp: int = Field(
        description=
        "relevant timestamp of scene source video, only provide an integer number"
    )


class HowToSteps(BaseModel):
    """Data model for how to step extracted information."""

    steps: List[HowToStep] = Field(
        description="List of steps to take to solve problem / accomplish task")
    summary: str = Field(
        description=
        "High level summary / commentary on how to solve problem / accomplish. task"
    )


@st.cache_resource
def load_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

    from llama_index.core import Settings
    Settings.embed_model = embed_model
    Settings.llm = llm

    vector_store = MilvusVectorStore(dim=EMBED_DIMS,
                                     uri=os.getenv('MILVUS_HOST'),
                                     token=os.getenv('MILVUS_TOKEN'),
                                     overwrite=False,
                                     collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store,
                                                   persist_dir=STORAGE)
    index = load_index_from_storage(storage_context=storage_context)

    query_engine = index.as_query_engine(
        output_cls=HowToSteps,
        response_mode=RESPONSE_MODE,
        llm=llm,
        verbose=True,
    )

    return query_engine, index.index_id


query_engine, index_id = load_query_engine()

# =====  Start Main Stream Lit App ===== #

with st.sidebar:
    st.title("Thousand Words Video Explorer")
    st.markdown(f"""
Loaded Index: `{STORAGE.split('/')[-1]}`

""")
    st.markdown("""
Info:

* Index contains ~212.7K YouTube videos that appeared to be english language and were in the "How To & Style" youtube category
* This simple video DB is representing over 9.4k hours of video (i.e. ~1.07 years of audio/visual information)

Example Queries:

* `how do you change a bicycle tire?`
* `show me how to boil an egg`
* `how do I grill a steak using charcoal?`

""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            response = message["content"]
            print(response)
            print(type(response.response))
            st.text("Here's what I found:")
            if str(type(response.response)) == "<class '__main__.HowToSteps'>":
                st.markdown(response.response.summary)
                for no, m in enumerate(response.response.steps):
                    st.header(f'Step {no}')
                    st.markdown(m.description)
                    if m.url:
                        split = m.url[len('https://www.youtube.com/watch?v='
                                          ):] + '_split_' + f'{m.scene:05d}'
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(
                                f"https://storage.googleapis.com/kdr-public/pandas70m/howto-travel/img/clip-start/{split}.jpg"
                            )
                            st.text("clip preview")
                        with col2:
                            st.video(m.url,
                                     format="video/mp4",
                                     start_time=m.timestamp)

            st.divider()
            vs = set([])
            t = "Sources:\n\n"
            for s in response.source_nodes:
                if s.metadata['video_url'] not in vs:
                    t += f"1. [{s.metadata['video_title']}]({s.metadata['video_url']})\n"
                    vs.add(s.metadata['video_url'])
            st.markdown(t)

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        print(response)
        print(type(response.response))
        st.text("Here's what I found:")
        if str(type(response.response)) == "<class '__main__.HowToSteps'>":
            st.markdown(response.response.summary)
            for no, m in enumerate(response.response.steps):
                st.header(f'Step {no}')
                st.markdown(m.description)
                if m.url:
                    split = m.url[len('https://www.youtube.com/watch?v='
                                      ):] + '_split_' + f'{m.scene:05d}'
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            f"https://storage.googleapis.com/kdr-public/pandas70m/howto-travel/img/clip-start/{split}.jpg"
                        )
                        st.text("clip preview")
                    with col2:
                        st.video(m.url,
                                 format="video/mp4",
                                 start_time=m.timestamp)
        st.divider()
        vs = set([])
        t = "Sources:\n\n"
        for s in response.source_nodes:
            if s.metadata['video_url'] not in vs:
                t += f"1. [{s.metadata['video_title']}]({s.metadata['video_url']})\n"
                vs.add(s.metadata['video_url'])
        st.markdown(t)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
