import math
import os
import os.path as osp
import random
from typing import Union

from SPARQLWrapper import JSON, SPARQLWrapper
from transformers.utils import get_json_schema


def tool(func):
    func._is_tool = True
    return func


class Tool:
    def __init__(self, functions: Union[str, list[str]]):
        self.tools_json = []

        if functions == "all":
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
                    self.tools_json.append(get_json_schema(attribute))
                else:
                    raise ValueError(f"Invalid function {fn_name}")
            else:
                raise ValueError(f"Unknown function {fn_name}")


class GraphDBTool(Tool):
    def __init__(self, functions: Union[str, list[str]], endpoint: str):
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

    def execute_query(self, query: str):
        self.wrapper.setQuery(query)
        return self.wrapper.query().convert()

    def id_in_graph(self, identifier: str):
        """Check if a given URI exists in some triple in the KG."""
        query = self._get_query(self.id_in_graph.__name__)
        query = query.replace("id0", identifier)
        result = self.execute_query(query)

        return result['boolean']

    @tool
    def get_entity_description(self, entity_id: str):
        """Retrieve description of an entity given its unique identifier.

        Args:
            entity_id: Identifier of the entity in the knowledge graph.
        """
        if not self.id_in_graph(entity_id):
            return f"Entity {entity_id} not found in knowledge graph."
        query = self._get_query(self.get_entity_description.__name__)
        query = query.replace("s0", entity_id)
        query_result = self.execute_query(query)["results"]["bindings"][0]
        output = dict()
        output[entity_id] = query_result["comment"]["value"]
        self.session_ids.add(entity_id)
        return output

    @tool
    def get_predicate_description(self, predicate_id: str):
        """Retrieve description of an predicate given its unique identifier.

        Args:
            predicate_id: Identifier of the predicate in the knowledge graph.
        """
        if not self.id_in_graph(predicate_id):
            return f"Predicate {predicate_id} not found in the graph."
        query = self._get_query(self.get_predicate_description.__name__)
        query = query.replace("s0", predicate_id)
        query_result = self.execute_query(query)["results"]["bindings"]

        output = dict()
        output[predicate_id] = ", ".join(d["label"]["value"] for d in query_result)

        self.session_ids.add(predicate_id)

        return output

    #@tool
    def search_entities(self, entity_query: str):
        """Find entity identifiers that best match a given search query.

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
            output[uri.split("/")[-1]] = comment

        if len(output) == 0:
            return "No matches found."
        else:
            return output

    #@tool
    def search_predicates(self, predicate_query: str):
        """Find predicate identifiers with a label matching a predicate
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

    def get_most_similar(self, unique_id: str):
        """Retrieve a list of entities or predicates that are semantically
        similar to a given entity or predicate identifier."""
        raise NotImplementedError

    def _run_predicate_query(self, query):
        query_result = self.execute_query(query)["results"]["bindings"]

        output = dict()
        for result in query_result:
            uri = result["id"]["value"]
            entity_id = uri.split("/")[-1]
            label = result["description"]["value"]
            output[entity_id] = label

            self.session_ids.add(entity_id)

        return output

    @tool
    def get_predicates_with_subject(self, entity_id: str):
        """Get a list of predicates in which the given entity occurs as a subject.

        Args:
            entity_id: the ID of the entity in the knowledge graph.
        """
        if not self.id_in_graph(entity_id):
            return f"Predicate {entity_id} not found in the graph."
        query = self._get_query(self.get_predicates_with_subject.__name__)
        query = query.replace("s0", entity_id)
        return self._run_predicate_query(query)

    @tool
    def get_predicates_with_object(self, entity_id: str):
        """Get a list of predicates in which the given entity occurs as a subject.

        Args:
            entity_id: the ID of the entity in the knowledge graph.
        """
        if not self.id_in_graph(entity_id):
            return f"Predicate {entity_id} not found in the graph."

        query = self._get_query(self.get_predicates_with_object.__name__)
        query = query.replace("s0", entity_id)
        return self._run_predicate_query(query)

    def _execute_count_distinct(self, query: str, predicate_id: str):
        query = query.replace("p0", predicate_id)
        query_result = self.execute_query(query)["results"]["bindings"][0]
        query_result = int(query_result["count"]["value"])

        return query_result

    def count_distinct_subjects(self, predicate_id: str):
        query = self._get_query(self.count_distinct_subjects.__name__)
        return self._execute_count_distinct(query, predicate_id)

    def count_distinct_objects(self, predicate_id: str):
        query = self._get_query(self.count_distinct_objects.__name__)
        return self._execute_count_distinct(query, predicate_id)

    def _execute_subject_or_object_query(self, query: str, predicate_id: str, num_results: int):
        if not self.id_in_graph(predicate_id):
            return f"Predicate {predicate_id} not found in the graph."

        page_size = 5
        num_pages = math.ceil(num_results / page_size)
        random_page_idx = random.randint(0, num_pages - 1)
        offset = random_page_idx * page_size
        query = query.replace("p0", predicate_id)
        query = query.replace("0000", f"{offset}")
        return self._run_predicate_query(query)

    @tool
    def get_subject_entities(self, predicate_id: str):
        """Get a random list of entities that occur as subjects of the given predicate identifier.

        Args:
            predicate_id: the ID of the predicate in the knowledge graph.
        """
        num_subjects = self.count_distinct_subjects(predicate_id)
        query = self._get_query(self.get_subject_entities.__name__)
        return self._execute_subject_or_object_query(query, predicate_id, num_subjects)

    @tool
    def get_object_entities(self, predicate_id: str):
        """Get a random list of entities that occur as objects of the given predicate identifier.

        Args:
            predicate_id: the ID of the predicate in the knowledge graph.
        """
        num_objects = self.count_distinct_objects(predicate_id)
        query = self._get_query(self.get_object_entities.__name__)
        return self._execute_subject_or_object_query(query, predicate_id, num_objects)


class AnswerStoreTool(Tool):
    def __init__(self):
        super().__init__(functions="all")
        self.answer = None

    def clear_answer(self):
        self.answer = None

    @tool
    def submit_final_answer(self, answer: str):
        """Submits the final answer to a user's question.

        Args:
            answer: Answer to be submitted.
        """
        self.answer = answer
        return "Answer submitted."


if __name__ == "__main__":
    from pprint import pprint

    tools = []
    for obj in dir(GraphDBTool):
        if hasattr(getattr(GraphDBTool, obj), "_is_tool"):
            tools.append(obj)
    db = GraphDBTool(tools, "http://localhost:7200/repositories/wikidata5m")
    # pprint(db.search_entities("michael jordan"))
    # pprint(db.get_entity_description("Q41421"))
    pprint(db.get_predicate_description("P31"))
    # pprint(db.search_predicates("capital of"))
    # pprint(db.get_predicates_with_object("Q41421"))
    # pprint(db.get_subject_entities("P3279"))
    # pprint(db.get_predicate_description("P3279"))
    # pprint(db.get_object_entities("P3279"))

    # import time
    #
    # start = time.time()
    # pprint(db.get_object_entities("P131"))
    # end = time.time()
    # print(f"Time taken: {end - start}")
