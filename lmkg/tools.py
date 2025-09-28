import os
import os.path as osp
import random
import socket
import urllib.error
from typing import Any, Callable

from SPARQLWrapper import JSON, SPARQLWrapper, SPARQLExceptions

from .exceptions import MalformedQueryException


def tool(func):
    func._is_tool = True
    return func


class Tool:
    def __init__(self, functions: list[str] = None):
        self.tools = []

        if functions is None:
            functions = []
            for obj_name in dir(self):
                obj = getattr(self, obj_name)
                if callable(obj) and hasattr(obj, "_is_tool"):
                    functions.append(obj_name)

        for fn_name in functions:
            if hasattr(self, fn_name):
                attribute = getattr(self, fn_name)
                is_tool = getattr(attribute, '_is_tool', False)
                if callable(attribute) and is_tool:
                    self.tools.append(attribute)
                else:
                    raise ValueError(f"Invalid function {fn_name}")
            else:
                raise ValueError(f"Unknown function {fn_name}")


class GraphDBTool(Tool):
    def __init__(self, endpoint: str, functions: list[str] = None):
        super().__init__(functions)
        self.wrapper = SPARQLWrapper(endpoint)
        self.wrapper.setReturnFormat(JSON)
        self.queries_dict = dict()
        self.session_ids = set()

    def _get_query(self, query_name: str):
        if query_name not in self.queries_dict:
            current_dir = osp.dirname(osp.abspath(__file__))
            query_path = osp.join(current_dir,
                                  "queries", f"{query_name}.sparql")
            if not os.path.exists(query_path):
                raise FileNotFoundError(f"Expected query file missing: "
                                        f"{query_path}")

            with open(query_path) as f:
                self.queries_dict[query_name] = f.read()
        return self.queries_dict[query_name]

    def clear_session_ids(self):
        self.session_ids = set()

    def is_alive(self):
        try:
            self.execute_query(self._get_query(self.is_alive.__name__))
            return True
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout, socket.error):
            return False

    def execute_query(self, query: str):
        try:
            self.wrapper.setQuery(query)
            results = self.wrapper.query().convert()
            return results
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout, socket.error) as e:
            raise ConnectionError(f"Connection failed: {e}") from e
        except SPARQLExceptions.QueryBadFormed as sparql_exception:
            raise MalformedQueryException(f"Attempted to run a malformed query. This is possibly due "
                                          f"to corrupted entity or predicate identifiers. Check the query:\n"
                                          f"{query}\n"
                                          f"Original error: {sparql_exception}")

    def check_id_in_graph(self, identifier: str):
        """Check if a given URI exists in some triple in the KG."""
        query = self._get_query(self.check_id_in_graph.__name__)
        query = query.replace("id0", identifier)
        result = self.execute_query(query)

        in_graph = result['boolean']
        if not in_graph:
            raise KeyError(f"{identifier} not in kg")

    def get_neighbors(self, identifier: str, identifier_pos: str, variable_pos: str):
        self.check_id_in_graph(identifier)

        query = self._get_query(self.get_neighbors.__name__)
        query = query.replace("?variable", f"?{variable_pos}").replace(f"?{identifier_pos}", f"wiki:{identifier}")
        results = self.execute_query(query)["results"]["bindings"]
        random.shuffle(results)

        output = []
        for result in results:
            uri = result[variable_pos]["value"]
            entity_id = uri.split("/")[-1]
            if entity_id.startswith("Q") or entity_id.startswith("P"):
                output.append(entity_id)
                self.session_ids.add(entity_id)
            if len(output) == 5:
                break

        description_predicate = "rdfs:label" if variable_pos == "p" else "rdfs:comment"
        max_length = 150 if variable_pos == "p" else 300
        return self.get_descriptions(output, description_predicate, check_in_graph=False, max_length=max_length)

    def get_descriptions(self, identifiers: list[str], predicate: str, check_in_graph: bool = True,
                         max_length: int = 300):
        if check_in_graph:
            for i in identifiers:
                self.check_id_in_graph(i)

        query = self._get_query(self.get_descriptions.__name__)
        query = query.replace("?predicate", f"{predicate}")
        values_list = '\n'.join([f"wiki:{i}" for i in identifiers])
        query = query.replace("values_list", values_list)
        query_results = self.execute_query(query)["results"]["bindings"]

        output = dict()
        for result in query_results:
            uri = result["id"]["value"]
            entity_id = uri.split("/")[-1]
            label = result["description"]["value"]

            if entity_id not in output:
                output[entity_id] = [label]
            else:
                output[entity_id].append(label)

        for identifier, descriptions in output.items():
            output[identifier] = ", ".join(descriptions)[:max_length]

        return output

    @tool
    def get_entity_description(self, entity_id: str):
        """Retrieve description of an entity given its unique KG identifier.

        Args:
            entity_id: Identifier of the entity in the knowledge graph.
        """
        return self.get_descriptions([entity_id], "rdfs:comment")

    @tool
    def get_predicate_description(self, predicate_id: str):
        """Retrieve description of a predicate given its unique KG identifier.

        Args:
            predicate_id: Identifier of the predicate in the knowledge graph.
        """
        return self.get_descriptions([predicate_id], "rdfs:label")

    @tool
    def search_entities(self, entity_query: str):
        """Find entity KG identifiers that best match a given search query.

        Args:
            entity_query: Entity query to search for.
        """
        query = self._get_query(self.search_entities.__name__)
        query = query.replace("q0", entity_query)
        query_results = self.execute_query(query)["results"]["bindings"]
        output = dict()
        for result in query_results:
            uri = result["e"]["value"]
            comment = result["shortComment"]["value"]
            entity_id = uri.split("/")[-1]
            self.session_ids.add(entity_id)
            output[entity_id] = comment

        if len(output) == 0:
            return "No matches found."
        else:
            return output

    @tool
    def search_predicates(self, predicate_query: str):
        """Find predicate KG identifiers with a label matching a predicate
        keyword.

        Args:
            predicate_query: Entity query to search for.
        """
        query = self._get_query(self.search_predicates.__name__)
        query = query.replace("q0", predicate_query)
        query_results = self.execute_query(query)["results"]["bindings"]

        predicate_labels = dict()
        for result in query_results:
            uri = result["e"]["value"]
            predicate_id = uri.split("/")[-1]
            self.session_ids.add(predicate_id)
            label = result["label"]["value"]
            if predicate_id not in predicate_labels:
                predicate_labels[predicate_id] = [label]
            else:
                predicate_labels[predicate_id].append(label)

        output = dict()
        for predicate, labels in predicate_labels.items():
            output[predicate] = ", ".join(labels)

        if len(output) == 0:
            return "No matches found."

        return output

    @tool
    def get_predicates_with_subject(self, entity_id: str):
        """Return a random list of predicates for which the given entity appears as the subject in the knowledge graph.

        Args:
            entity_id: the ID of the entity in the knowledge graph.
        """
        return self.get_neighbors(entity_id, "s", "p")

    @tool
    def get_predicates_with_object(self, entity_id: str):
        """Return a random list of predicates for which the given entity appears as the object in the knowledge graph.

        Args:
            entity_id: the ID of the entity in the knowledge graph.
        """
        return self.get_neighbors(entity_id, "o", "p")

    @tool
    def get_subject_entities(self, predicate_id: str):
        """Return a random list of entities that appear as subjects in triples with the given predicate.

        Args:
            predicate_id: the ID of the predicate in the knowledge graph.
        """
        return self.get_neighbors(predicate_id, "p", "s")

    @tool
    def get_object_entities(self, predicate_id: str):
        """Return a random list of entities that appear as objects in triples with the given predicate.

        Args:
            predicate_id: the ID of the predicate in the knowledge graph.
        """
        return self.get_neighbors(predicate_id, "p", "o")


class AnswerStoreTool(Tool):
    def __init__(self, graphdb_tool: GraphDBTool, answer_parser: Callable[[str], Any] = None):
        super().__init__()
        self.answer = None
        self.graphdb = graphdb_tool
        self.initial_ids = None
        self.answer_parser = answer_parser

    def initialize(self, initial_ids: set[str] = None):
        self.answer = None
        self.initial_ids = initial_ids

    @tool
    def submit_final_answer(self, answer: str):
        """Submits the final answer to a user's question.

        Args:
            answer: Answer to be submitted.
        """
        return_string = "Answer submitted"
        if self.answer_parser:
            try:
                self.answer, ids_in_answer = self.answer_parser(answer)
                valid_ids = self.graphdb.session_ids.union(self.initial_ids if self.initial_ids else set())
                hallucinated_ids = ids_in_answer.difference(valid_ids)
                if hallucinated_ids:
                    return_string = (f"The answer contains identifiers that were not retrieved "
                                     f"by any function: {', '.join(hallucinated_ids)}. Please try again.")
            except Exception as e:
                return_string = f"Error parsing answer: {e}"
        else:
            self.answer = answer

        return return_string


if __name__ == "__main__":
    from pprint import pprint

    db = GraphDBTool(functions="all", endpoint="http://localhost:7200/repositories/wikidata5m")
    # pprint(db.search_entities("michael jordan"))
    # pprint(db.search_predicates("capital"))
    #
    # pprint(db.get_entity_description("Q55"))
    # pprint(db.get_predicate_description("P31"))

    pprint(db.get_predicates_with_subject("Q55"))
    pprint(db.get_predicates_with_object("Q55"))
    #
    pprint(db.get_subject_entities("P131"))
    pprint(db.get_object_entities("P131"))
