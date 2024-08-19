from SPARQLWrapper import SPARQLWrapper, JSON


class GraphDBConnector:
    def __init__(self, endpoint: str):
        self.wrapper = SPARQLWrapper(endpoint)
        self.wrapper.setReturnFormat(JSON)

    def execute(self, query: str):
        self.wrapper.setQuery(query)
        return self.wrapper.query().convert()

    def id_in_graph(self, identifier: str):
        """Check if a given URI exists in some triple in the KG."""
        with open("queries/id_in_graph.sparql") as f:
            query = f.read()
        query = query.replace("s0", identifier)
        result = self.execute(query)

        return result['boolean']

    def get_definition(self, unique_id: str):
        """Retrieve textual definition or label of an entity or predicate given
        its unique identifier."""
        if not self.id_in_graph(unique_id):
            raise ValueError(f"Identifier {unique_id} not found in the graph.")

        with open("queries/get_definition.sparql") as f:
            query = f.read()
        query = query.replace("s0", unique_id)
        return self.execute(query)

    def search_entities(self, entity_query: str):
        """Search for entities with a label matching the given query."""
        with open("queries/search_entities.sparql") as f:
            query = f.read()
        query = query.replace("q0", entity_query)
        return self.execute(query)

    def search_predicates(self, predicate_query: str):
        """Search for predicates with a label matching the given query."""
        with open("queries/search_predicates.sparql") as f:
            query = f.read()
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
    print(db.search_entities("michael jordan"))
