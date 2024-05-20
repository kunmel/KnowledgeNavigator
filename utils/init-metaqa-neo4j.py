from py2neo import Graph

MetaQA = "path to kb.txt"
graph=Graph("url2neo4j",auth=("username", "password"),name="neo4j")

tail_label_dict = {
    'directed_by': "director",
    'written_by': "writer",
    'starred_actors': "actor",
    'release_year': "year",
    'in_language': "language",
    'has_tags': "tag",
    'has_genre': "genre",
    'has_imdb_votes': "imdbvotes",
    'has_imdb_rating': "imdbrating"
}

with open(MetaQA, "r") as f:
    lines = f.readlines()
    relations = []
    for line in lines:
        triple = line.strip().split("|")
        cql = """MERGE (head:Movie {{name: "{movie_name}"}}) MERGE (tail:{tail_label} {{name: "{tail_name}"}}) MERGE (head)-[:{realtion}]->(tail) """.format(movie_name=triple[0], tail_label=tail_label_dict[triple[1]], tail_name=triple[2], realtion=triple[1])
        graph.run(cql)
    