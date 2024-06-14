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

STORAGE = "./storage/travel-69k-bge-small-en-384"
EMBED = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384
RESPONSE_MODE = "tree_summarize"
COLLECTION = "travel69k"


class NotablePlaceMention(BaseModel):
    """Notable place mention data model"""

    name: str = Field(description="the place name")
    description: str = Field(
        description=
        "one to two sentence description of the place and why it is well known / why people love it"
    )
    place_type: str = Field(
        description=
        "type of place / point of interest, e.g. restaurant, landmark, transportation, shopping, etc."
    )
    best_known_for: List[str] = Field(
        description="list of things this place is know for")
    url: str = Field(description="link to the source youtube video")
    scene: int = Field(
        description=
        "relevant video scene number in source video, only provide an integer number"
    )
    timestamp: int = Field(
        description=
        "relevant timestamp of scene source video, only provide an integer number"
    )


class NotablePlaceMentionsSummary(BaseModel):
    """Data model for notable place mentions extracted information."""

    place_mentions: List[NotablePlaceMention] = Field(
        description="List of notable places mentioned in retrieved results")
    summary: str = Field(
        description=
        "High level summary / commentary on the places retrieved and how relevant to the query"
    )


@st.cache_resource
def load_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)

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
        output_cls=NotablePlaceMentionsSummary,
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

* Index contains ~69.3K YouTube videos that appeared to be english language and were in the "Travel & Events" youtube category
* This simple video DB is representing over 3k hours of video (i.e. ~127 days of audio/visual information)

Example Queries:

* `what is the best museum in paris?`
* `where can i find the best pizza in nyc?`
* `what are the top tourist attractions in vietnam?`

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
            if str(type(response.response)
                   ) == "<class '__main__.NotablePlaceMentionsSummary'>":
                st.markdown(response.response.summary)
                for m in response.response.place_mentions:
                    st.header(m.name)
                    t = m.description + "\n\n**Known For**:\n"
                    for b in m.best_known_for:
                        t += "\n* " + b
                    st.markdown(t)
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
        if str(type(response.response)
               ) == "<class '__main__.NotablePlaceMentionsSummary'>":
            st.markdown(response.response.summary)
            for m in response.response.place_mentions:
                st.header(m.name)
                t = m.description + "\n\n**Known For**:\n"
                for b in m.best_known_for:
                    t += "\n* " + b
                st.markdown(t)
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
