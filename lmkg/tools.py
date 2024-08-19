from SPARQLWrapper import SPARQLWrapper, JSON
import functools


def load_sparql_query(func: callable):
    """Decorator to load a SPARQL query file based on a method name."""
    queries_dict = {}

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        method_name = func.__name__

        if method_name not in queries_dict:
            with open(f"queries/{method_name}.sparql", "r") as f:
                print(f"Loading SPARQL query for {method_name}")
                queries_dict[method_name] = f.read()

        return func(self, queries_dict[method_name], *args, **kwargs)

    return wrapper


class GraphDBConnector:
    def __init__(self, endpoint: str):
        self.wrapper = SPARQLWrapper(endpoint)
        self.wrapper.setReturnFormat(JSON)

    def execute(self, query: str):
        self.wrapper.setQuery(query)
        return self.wrapper.query().convert()

    @load_sparql_query
    def id_in_graph(self, query: str, identifier: str):
        """Check if a given URI exists in some triple in the KG."""
        query = query.replace("s0", identifier)
        result = self.execute(query)

        return result['boolean']

    @load_sparql_query
    def get_definition(self, query: str, unique_id: str):
        """Retrieve textual definition or label of an entity or predicate given
        its unique identifier."""
        if not self.id_in_graph(unique_id):
            raise ValueError(f"Identifier {unique_id} not found in the graph.")
        query = query.replace("s0", unique_id)
        return self.execute(query)

    @load_sparql_query
    def search_entities(self, query: str, entity_query: str):
        """Search for entities with a label matching the given query."""
        query = query.replace("q0", entity_query)
        return self.execute(query)

    @load_sparql_query
    def search_predicates(self, query: str, predicate_query: str):
        """Search for predicates with a label matching the given query."""
        query = query.replace("q0", predicate_query)
        return self.execute(query)

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
