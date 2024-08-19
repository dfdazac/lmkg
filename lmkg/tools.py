from SPARQLWrapper import JSON, SPARQLWrapper
from transformers.utils import get_json_schema


def tool(func):
    func._is_tool = True
    return func


class GraphDBConnector:
    def __init__(self, endpoint: str):
        self.wrapper = SPARQLWrapper(endpoint)
        self.wrapper.setReturnFormat(JSON)
        self.queries_dict = dict()

        self.tools_json = []
        for attr_name in dir(self):
            attribute = getattr(self, attr_name)
            if callable(attribute) and getattr(attribute, '_is_tool', False):
                self.tools_json.append(get_json_schema(attribute))

    def _get_query(self, query_name: str):
        if query_name not in self.queries_dict:
            with open(f"queries/{query_name}.sparql", "r") as f:
                self.queries_dict[query_name] = f.read()
        return self.queries_dict[query_name]

    def execute_query(self, query: str):
        self.wrapper.setQuery(query)
        return self.wrapper.query().convert()

    def id_in_graph(self, identifier: str):
        """Check if a given URI exists in some triple in the KG."""
        query = self._get_query(self.id_in_graph.__name__)
        query = query.replace("s0", identifier)
        result = self.execute_query(query)

        return result['boolean']

    @tool
    def get_definition(self, unique_id: str):
        """Retrieve definition of an entity or predicate given its unique identifier.

        Args:
            unique_id: Unique identifier of an entity or predicate.
        """
        if not self.id_in_graph(unique_id):
            raise ValueError(f"Identifier {unique_id} not found in the graph.")
        query = self._get_query(self.get_definition.__name__)
        query = query.replace("s0", unique_id)
        return self.execute_query(query)

    @tool
    def search_entities(self, entity_query: str):
        """Find entity identifiers that best match a given search query.

        Args:
            entity_query: Entity query to search for.
        """
        query = self._get_query(self.search_entities.__name__)
        query = query.replace("q0", entity_query)
        return self.execute_query(query)

    def search_predicates(self, predicate_query: str):
        """Find predicate identifiers that best match a given search query.

        Args:
            predicate_query: Entity query to search for.
        """
        query = self._get_query(self.search_predicates.__name__)
        query = query.replace("q0", predicate_query)
        return self.execute_query(query)

    def get_most_similar(self, unique_id: str):
        """Retrieve a list of entities or predicates that are semantically
        similar to a given entity or predicate identifier."""
        raise NotImplementedError

    def get_predicates_with_subject(self, predicate_id: str):
        """Get a random list of predicates in which the given entity occurs as
        a subject.

        Args:
            predicate_id: the ID of the
        """
        raise NotImplementedError

    def get_predicates_with_object(self, predicate_id: str):
        """Get a random list of predicates in which the given entity occurs as
        an object."""
        raise NotImplementedError

    def get_subject_entities(self, entity_id: str):
        """Get a random list of predicates in which the given entity occurs as
        a subject."""
        raise NotImplementedError

    def get_object_entities(self, entity_id: str):
        """Get a random list of predicates in which the given entity occurs as
        an object."""
        raise NotImplementedError


if __name__ == "__main__":
    db = GraphDBConnector("http://localhost:7200/repositories/wikidata5m")
    db.search_entities("michael jordan")
    db.search_entities("michael jordan")
    db.search_entities("michael jordan")
    db.search_entities("alma mater")
    db.search_predicates("alma mater")
    db.search_predicates("alma mater")
    db.search_entities("michael jordan")
