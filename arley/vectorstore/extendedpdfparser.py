import openparse
from openparse import Node
from openparse import processing
# import numpy as np

# https://github.com/Filimoa/open-parse/issues/30#issuecomment-2073595236



# https://github.com/miku/open-parse/blob/miku/ollama-embeddings/src/openparse/processing/semantic_transforms.py
# class OllamaEmbeddings:
#     """
#     Use local models via ollama for calculating embeddings. Uses the REST API
#     https://github.com/ollama/ollama/blob/main/docs/api.md.
#
#     * nomic-embed-text
#     * mxbai-embed-large
#     """
#
#     def __init__(
#         self,
#         url: str = "http://localhost:11434/",
#         model: str = "mxbai-embed-large",
#         batch_size: int = 256,
#     ):
#         """
#         Used to generate embeddings for Nodes.
#         """
#         self.url = url
#         self.model = model
#         self.batch_size = batch_size
#
#     def embed_many(self, texts: List[str]) -> List[List[float]]:
#         """
#         Generate embeddings for a list of texts. Support for batches coming
#         soon, cf. https://ollama.com/blog/embedding-models
#
#         Args:
#             texts (list[str]): The list of texts to embed.
#             batch_size (int): The number of texts to process in each batch.
#
#         Returns:
#             List[List[float]]: A list of embeddings.
#         """
#         conn = self._create_conn()
#         res = []
#         for i in range(0, len(texts), self.batch_size):
#             batch_texts = texts[i : i + self.batch_size]
#             for text in batch_texts:
#                 params = json.dumps({"model": self.model, "prompt": text})
#                 headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
#                 conn.request("POST", "/api/embeddings", params, headers)
#                 response = conn.getresponse()
#                 if response.status != 200:
#                     raise RuntimeError(
#                         "embeddings request failed: {} {}".format(
#                             response.status, response.reason
#                         )
#                     )
#                 doc = json.loads(response.read())
#                 res.extend(doc["embedding"])
#         conn.close()
#         return res
#
#     def _create_conn(self):
#         parsed = urlparse(self.url)
#         if parsed.scheme == "https":
#             return HTTPSConnection(parsed.hostname, parsed.port)
#         else:
#             return HTTPConnection(parsed.hostname, parsed.port)


# https://github.com/miku/open-parse/blob/miku/ollama-embeddings/src/openparse/processing/ingest.py
# class LocalSemanticIngestionPipeline(IngestionPipeline):
#     """
#     A semantic pipeline for ingesting and processing Nodes using ollama for embeddings.
#     """
#
#     def __init__(
#         self,
#         url: str = "http://localhost:11434",
#         model: str = "mxbai-embed-large",
#         min_tokens: int = consts.TOKENIZATION_LOWER_LIMIT,
#         max_tokens: int = consts.TOKENIZATION_UPPER_LIMIT,
#     ) -> None:
#         embedding_client = OllamaEmbeddings(url=url, model=model)
#
#         self.transformations = [
#             RemoveTextInsideTables(),
#             RemoveFullPageStubs(max_area_pct=0.35),
#             # mostly aimed at combining bullets and weird formatting
#             CombineNodesSpatially(
#                 x_error_margin=10,
#                 y_error_margin=2,
#                 criteria="both_small",
#             ),
#             CombineHeadingsWithClosestText(),
#             CombineBullets(),
#             RemoveMetadataElements(),
#             RemoveRepeatedElements(threshold=2),
#             RemoveNodesBelowNTokens(min_tokens=10),
#             CombineBullets(),
#             CombineNodesSemantically(
#                 embedding_client=embedding_client,
#                 min_similarity=0.6,
#                 max_tokens=max_tokens // 2,
#             ),
#             CombineNodesSemantically(
#                 embedding_client=embedding_client,
#                 min_similarity=0.55,
#                 max_tokens=max_tokens,
#             ),
#             RemoveNodesBelowNTokens(min_tokens=min_tokens),
#         ]
#
# def cosine_similarity(
#     a: np.ndarray | list[float], b: np.ndarray | list[float]
# ) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
#
# def get_node_similarities(nodes: list[Node]):
#     # get the similarity of each node with the node that precedes it
#     embeddings = embedding_client.embed_many([node.text for node in nodes])
#     similarities = []
#     for i in range(1, len(embeddings)):
#         similarities.append(cosine_similarity(embeddings[i - 1], embeddings[i]))
#
#     similarities = [round(sim, 2) for sim in similarities]
#     return [0] + similarities
#
#
#
# def parse_pdf(file: Path):
#     parser: openparse.DocumentParser = openparse.DocumentParser()
#     parsed_basic_doc: openparse.schemas.ParsedDocument = parser.parse(file)
#
#     for node in parsed_basic_doc.nodes:
#         node_data: dict = node.model_dump(mode="json", by_alias=True, exclude_none=True)
#         logger.debug(Helper.get_pretty_dict_json_no_sort(node_data))
#
#     lala = openparse.processing.SemanticIngestionPipeline()
#     # from openparse import processing, DocumentParser
#     #
#     # semantic_pipeline = processing.SemanticIngestionPipeline(
#     #     openai_api_key=OPEN_AI_KEY,
#     #     model="text-embedding-3-large",
#     #     min_tokens=64,
#     #     max_tokens=1024,
#     # )
#     # parser = DocumentParser(
#     #     processing_pipeline=semantic_pipeline,
#     # )
#     # parsed_content = parser.parse(basic_doc_path)
#
#     return 0