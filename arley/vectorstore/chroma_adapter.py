import textwrap
from io import StringIO
from typing import Self, Sequence, Tuple, Mapping, Set, Literal, Any

from arley import Helper
from arley.config import settings, is_in_cluster

from loguru import logger

from arley.Helper import Singleton, get_pretty_dict_json_no_sort

import os
import chromadb

from chromadb.api.models.Collection import Collection as ChromaCollection

from arley.dbobjects.ragdoc import DocTypeEnum
from arley.llm import ollama_adapter

#from chromadb import Settings, ClientAPI


# https://docs.trychroma.com/guides
# https://docs.trychroma.com/reference/py-client

assert settings.chromadb.ollama_embed_model is not None and settings.chromadb.default_collectionname is not None

_CHROMADB_OLLAMA_DEFAULT_EMBED_MODEL: str = settings.chromadb.ollama_embed_model   # "nomic-embed-text:latest"  # "mxbai-embed-large:latest"  # "nomic-embed-text:latest"
_CHROMADB_DEFAULT_COLLECTION_NAME: str = settings.chromadb.default_collectionname

from chromadb import Documents, EmbeddingFunction, Embeddings, QueryResult, GetResult, SparseVector


class MyEmbeddingFunction(EmbeddingFunction):
    from chromadb.utils import embedding_functions
    _default_ef = embedding_functions.DefaultEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        # logger.debug(f"{type(input)=} {len(input)=} {input=} ")

        embedding_list: Embeddings = []

        for doc in input:
            embeddings: Sequence[float|int]  = ollama_adapter.embeddings(
                embed_model=_CHROMADB_OLLAMA_DEFAULT_EMBED_MODEL,
                prompt=doc,
                # num_ctx=None  # if different from nomic-embed, this has to be checked!
            )
            embedding_list.append(embeddings)  # type: ignore


        return embedding_list


# does not necessarily need to be a singleton...
class ChromaDBConnection(metaclass=Singleton):
    logger = logger.bind(classname=__qualname__)

    def __init__(self) -> None:
        super().__init__()

        self._CHROMADB_HOST: str = os.getenv("CHROMADB_HOST", settings.chromadb.host)
        self._CHROMADB_PORT: int = int(os.getenv("CHROMADB_PORT", "80"))

        if is_in_cluster():
            assert settings.chromadb.host_in_cluster is not None and settings.chromadb.port is not None
            self._CHROMADB_HOST = settings.chromadb.host_in_cluster
            self._CHROMADB_PORT = settings.chromadb.port

        self.logger.debug(f"CHROMADB_HOST: {self._CHROMADB_HOST}")
        self.logger.debug(f"CHROMADB_PORT: {self._CHROMADB_PORT}")

        self._chromadb_settings: chromadb.Settings | None = None

        if settings.chromadb.http_auth_user and settings.chromadb.http_auth_pass:
            self._chromadb_settings = chromadb.Settings(
                chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                chroma_client_auth_credentials=f"{settings.chromadb.http_auth_user}:{settings.chromadb.http_auth_pass}"
            )

        self.chroma_client: chromadb.ClientAPI = chromadb.HttpClient(
            host=self._CHROMADB_HOST,
            port=self._CHROMADB_PORT,
            settings=self._chromadb_settings
        )


    def get_client(self) -> chromadb.ClientAPI:
        return self.chroma_client


    def get_or_create_collection(self, collectionname: str) -> ChromaCollection:
        cdbcollection: ChromaCollection = self.get_client().get_or_create_collection(
            name=collectionname,
            metadata={"hnsw:space": "cosine"},
            embedding_function=MyEmbeddingFunction()
        )
        
        return cdbcollection

    @classmethod
    def get_context_augmentations(cls,
                                  prompt: str,
                                  cdbcollection: ChromaCollection,
                                  initial_topic: str|None = None,
                                  lang: Literal["de", "en"]|None = None,
                                  n_results: int = 10,
                                  only_contracts: bool = False) -> dict[str, list[dict]]:
        logger = cls.logger

        ret: dict[str, list[dict]] = dict()

        logger.debug(f"{type(cdbcollection)=}")

        qt: str = prompt
        if initial_topic:
            qt = initial_topic.strip() + "\n" + prompt

        where: dict|None = None
        where_and: list[dict] = []

        if only_contracts:
            # where = {"$or": [{"doctype": DocTypeEnum.contract.value}, {"doctype": DocTypeEnum.title_lookup.value}]})
            where_doctype: dict = {"doctype": {"$in": [
                DocTypeEnum.contract.value,
                DocTypeEnum.targeted_by_prompts_lookup.value,
                DocTypeEnum.title_lookup.value,
                DocTypeEnum.categorization_lookup.value
            ]}}
            where_and.append(where_doctype)

        if lang:
            where_lang: dict = {
                "lang": lang
            }
            where_and.append(where_lang)

        match(len(where_and)):
            case 0:
                ...
            case 1:
                where = where_and[0]
            case _:
                where = {
                    "$and": where_and
                }

        logger.debug(f"WHERE-DOC:\n{Helper.get_pretty_dict_json_no_sort(where)}")

        qr: QueryResult = cdbcollection.query(
            query_texts=[qt],
            n_results=n_results,
            where=where
        )

        included_contract_ids: dict[str, dict] = dict()

        for mode in ["lookup", "normal"]:
            for ind, myid in enumerate(qr["ids"][0]):
                assert qr["distances"] is not None and qr["metadatas"] is not None

                distance: float = qr["distances"][0][ind]
                metadata: Mapping[str, str | int | float | bool | SparseVector | None] = qr["metadatas"][0][ind]

                assert qr["documents"] is not None
                document: str = qr["documents"][0][ind]

                dt = metadata["doctype"]

                if mode == "lookup":
                    if dt is not None and isinstance(dt, str) and not dt.endswith("_lookup"):
                        if not dt in ret:  # ret.keys():
                            ret[dt] = []
                        continue

                if mode == "normal" and dt is not None and isinstance(dt, str) and dt.endswith("_lookup"):
                    continue

                assert isinstance(dt, str)

                medict: dict = dict()

                medict["id"] = myid
                medict["index"] = ind
                medict["distance"] = distance
                medict["metadata"] = dict(metadata)
                medict["document"] = document
                medict["foundby"] = []

                main_contract_id: str | int | float | bool | SparseVector | None = metadata.get("parent_id")
                if not main_contract_id:
                    main_contract_id = myid

                assert main_contract_id is not None and isinstance(main_contract_id, str)

                medict["main_contractid"] = main_contract_id

                match metadata["doctype"]:
                    case "contract":
                        contractdict: dict = medict
                        if myid not in included_contract_ids:  #.keys():
                            ret[metadata["doctype"]].append(medict)
                            included_contract_ids[myid] = medict
                        else:
                            contractdict = included_contract_ids[myid]

                        contractdict["distance"] = min(contractdict["distance"], distance)
                        contractdict["foundby"].append({"doctype": metadata["doctype"], "id": myid, "distance": distance})
                    case s if s is not None and isinstance(s, str) and s.endswith("_lookup"):
                        if main_contract_id not in included_contract_ids:  #.keys():
                            main_doc, main_doc_metadata = ChromaDBConnection.get_document_by_id(doc_id=main_contract_id, cdbcollection=cdbcollection)
                            main_doc_dict: dict = dict()

                            main_doc_dict["id"] = main_contract_id
                            main_doc_dict["index"] = ind
                            main_doc_dict["distance"] = distance
                            main_doc_dict["metadata"] = main_doc_metadata  # no need to clone/copy
                            main_doc_dict["document"] = main_doc
                            main_doc_dict["foundby"] = []

                            main_doc_dict["foundby"].append({"doctype": metadata["doctype"], "id": myid, "distance": distance})

                            included_contract_ids[main_contract_id] = main_doc_dict
                            if not "contract" in ret:  #.keys():
                                ret["contract"] = []
                            ret["contract"].append(main_doc_dict)
                        else:
                            contractdict = included_contract_ids[main_contract_id]
                            contractdict["foundby"].append({"doctype": metadata["doctype"], "id": myid, "distance": distance})
                            contractdict["distance"] = min(contractdict["distance"], distance)
                    case _:
                        ret[dt].append(medict)

        return ret

    @staticmethod
    def add_document(doc_id: str, document: str, metadata: dict, cdbcollection: ChromaCollection) -> None:
        cdbcollection.add(ids=[doc_id],
                       # embeddings=,
                       metadatas=[metadata],
                       documents=[document]
                       )

    @staticmethod
    def get_document_by_id(doc_id: str,
                     cdbcollection: ChromaCollection) -> Tuple[str, Mapping[str, str | int | float | bool | SparseVector | None] | Any]:
        gq: GetResult = cdbcollection.get(doc_id)

        docs = gq["documents"]
        metas = gq["metadatas"]

        assert docs is not None and metas is not None

        return docs[0], metas[0]


    @classmethod
    def get_instance(cls) -> "ChromaDBConnection":
        return ChromaDBConnection()  # is singleton



# import ollama, chromadb, time
# from utilities import readtext, getconfig
# from mattsollamatools import chunker, chunk_text_by_sentences
#
# collectionname="buildragwithpython"
#
# chroma = chromadb.HttpClient(host="localhost", port=8000)
# print(chroma.list_collections())
# if any(collection.name == collectionname for collection in chroma.list_collections()):
#   print('deleting collection')
#   chroma.delete_collection("buildragwithpython")
# collection = chroma.get_or_create_collection(name="buildragwithpython", metadata={"hnsw:space": "cosine"})
#
# embedmodel = getconfig()["embedmodel"]
# starttime = time.time()
# with open('sourcedocs.txt') as f:
#   lines = f.readlines()
#   for filename in lines:
#     text = readtext(filename)
#     chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0 )
#     print(f"with {len(chunks)} chunks")
#     for index, chunk in enumerate(chunks):
#       embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
#       print(".", end="", flush=True)
#       collection.add([filename+str(index)], [embed], documents=[chunk], metadatas={"source": filename})
#
# print("--- %s seconds ---" % (time.time() - starttime))


# collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
# collection = client.get_collection(name="my_collection", embedding_function=emb_fn)
# If you later wish to get_collection, you MUST do so with the embedding function you supplied while creating the collection

# collection = client.get_collection(name="test") # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.
# collection = client.get_or_create_collection(name="test") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
# client.delete_collection(name="my_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
# Collections have a few useful convenience methods.
#
# Copy Code
# python
#
# collection.peek() # returns a list of the first 10 items in the collection
# collection.count() # returns the number of items in the collection
# collection.modify(name="new_name") # Rename the collection

#  collection = client.create_collection(
#         name="collection_name",
#         metadata={"hnsw:space": "cosine"} # l2 is the default
#     )
#
# https://docs.trychroma.com/guides#changing-the-distance-function
# Valid options for hnsw:space are "l2", "ip, "or "cosine". The default is "l2" which is the squared L2 norm.
# ip: inner product

# collection.add(
#     documents=["lorem ipsum...", "doc2", "doc3", ...],
#     metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
#     ids=["id1", "id2", "id3", ...]
# )

# collection.add(
#     documents=["doc1", "doc2", "doc3", ...],
#     embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
#     metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
#     ids=["id1", "id2", "id3", ...]
# )

# collection.query(
#     query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
#     n_results=10,
#     where={"metadata_field": "is_equal_to_this"},
#     where_document={"$contains":"search_string"}
# )


# collection.query(
#     query_texts=["doc10", "thus spake zarathustra", ...],
#     n_results=10,
#     where={"metadata_field": "is_equal_to_this"},
#     where_document={"$contains":"search_string"}
# )


# import ollama, sys, chromadb
# from utilities import getconfig
#
# embedmodel = getconfig()["embedmodel"]
# mainmodel = getconfig()["mainmodel"]
# chroma = chromadb.HttpClient(host="localhost", port=8000)
# collection = chroma.get_or_create_collection("buildragwithpython")
#
# query = " ".join(sys.argv[1:])
# queryembed = ollama.embeddings(model=embedmodel, prompt=query)["embedding"]
#
#
# relevantdocs = collection.query(query_embeddings=[queryembed], n_results=5)["documents"][0]
# docs = "\n\n".join(relevantdocs)
# modelquery = f"{query} - Answer that question using the following text as a resource: {docs}"
#
# stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)
#
# for chunk in stream:
#     if chunk["response"]:
#         print(chunk["response"], end="", flush=True)

def main_old() -> None:
    collectionname: str = _CHROMADB_DEFAULT_COLLECTION_NAME
    chromadbconn: ChromaDBConnection = ChromaDBConnection.get_instance()

    # chromadbconn.get_client().delete_collection(collectionname)

    logger.debug(chromadbconn.get_client().heartbeat())  # this should work with or without authentication - it is a public endpoint
    logger.debug(chromadbconn.get_client().get_version())  # this should work with or without authentication - it is a public endpoint
    logger.debug(chromadbconn.get_client().list_collections())  # this is a protected endpoint and requires authentication

    collection: chromadb.api.Collection = chromadbconn.get_or_create_collection(collectionname)

    logger.debug(f"{type(collection)=}")

    qg: GetResult = collection.get("2b0773d2-c3ee-4d29-b564-90941efe99c5")
    logger.debug(get_pretty_dict_json_no_sort(qg))

    for ind, myid in enumerate(qg["ids"]):
        logger.debug("\n")
        logger.debug(get_pretty_dict_json_no_sort(myid))

        metas = qg["metadatas"]
        docs = qg["documents"]

        assert metas is not None and docs is not None

        logger.debug(get_pretty_dict_json_no_sort(metas[ind]))
        logger.debug(get_pretty_dict_json_no_sort(docs[ind]))

    qr: QueryResult = collection.query(
        query_texts=["Erstelle mir ein NDA"],
        n_results=10,
        # where={
        #     "$and": [{"doctype": "contract"}, {"id": "2b0773d2-c3ee-4d29-b564-90941efe99c5"}]
        # },
        # where_document={"$contains":"search_string"}
    )

    # logger.debug(get_pretty_dict_json_no_sort(qr["ids"]))
    # logger.debug(get_pretty_dict_json_no_sort(qr["distances"]))
    # logger.debug(get_pretty_dict_json_no_sort(qr["metadatas"]))
    # logger.debug(get_pretty_dict_json_no_sort(qr["documents"]))

    for ind, myid in enumerate(qr["ids"][0]):
        logger.debug("\n")
        logger.debug(get_pretty_dict_json_no_sort(myid))
        logger.debug(get_pretty_dict_json_no_sort(qr["distances"][0][ind]))  # type: ignore
        logger.debug(get_pretty_dict_json_no_sort(qr["metadatas"][0][ind]))  # type: ignore
        logger.debug(get_pretty_dict_json_no_sort(qr["documents"][0][ind]))  # type: ignore

    #
# collection.query(
#     query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
#     n_results=10,
#     where={"metadata_field": "is_equal_to_this"},
#     where_document={"$contains":"search_string"}
# )


# collection.query(
#     query_texts=["doc10", "thus spake zarathustra", ...],
#     n_results=10,
#     where={"metadata_field": "is_equal_to_this"},
#     where_document={"$contains":"search_string"}
# )


# import ollama, sys, chromadb
# from utilities import getconfig
#
# embedmodel = getconfig()["embedmodel"]
# mainmodel = getconfig()["mainmodel"]
# chroma = chromadb.HttpClient(host="localhost", port=8000)
# collection = chroma.get_or_create_collection("buildragwithpython")
#
# query = " ".join(sys.argv[1:])
# queryembed = ollama.embeddings(model=embedmodel, prompt=query)["embedding"]
#
#
# relevantdocs = collection.query(query_embeddings=[queryembed], n_results=5)["documents"][0]
# docs = "\n\n".join(relevantdocs)
# modelquery = f"{query} - Answer that question using the following text as a resource: {docs}"
#
# stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)
#
# for chunk in stream:
#     if chunk["response"]:
#         print(chunk["response"], end="", flush=True)

def main() -> None:
    ...

if __name__ == "__main__":
  main()