import config
import re
from collections import defaultdict

def decode_similar_queries(text):
    regular_expression = r'(?:'
    for i in range(config.SIMILAR_QUERY_GENERATION_COUNT):
        head = "Q"+str(i+1)
        if head not in text:
            return False, []
        regular_expression = regular_expression + head + ":"
        if i != config.SIMILAR_QUERY_GENERATION_COUNT - 1:
            regular_expression = regular_expression + "|"
    regular_expression = regular_expression + r")\s+(.*?)(?=\n|$)"
    return True, re.findall(regular_expression, text)

def decode_relation_retrieve(text):
    regular_expression = r'(?:'
    for i in range(config.LLM_RELATION_CHOOSE_NUMBER):
        head = "R"+str(i+1)
        if head not in text:
            return False, []
        regular_expression = regular_expression + head + ":"
        if i != config.LLM_RELATION_CHOOSE_NUMBER - 1:
            regular_expression = regular_expression + "|"
    regular_expression = regular_expression + r")\s+(.*?)(?=\n|$)"
    return True, re.findall(regular_expression, text)


def triple2text(triples):
    knowledge_str = ""
    relation_head_tail_dict =  {
        "directed_by": ["movie", "director"],
        "has_genre": ["movie", "genre"],
        "has_imdb_rating": ["movie", "imdb rating"],
        "has_imdb_votes": ["movie", "imdb votes"],
        "has_tags": ["movie", "tags"],
        "in_language": ["movie", "language"],
        "release_year": ["movie", "release year"],
        "starred_actors": ["movie", "actor"],
        "written_by": ["movie", "writter"]
    }

    movie_head_triple_dict = defaultdict(list)
    movie_tail_triple_dict = defaultdict(list)

    for triple in triples:
        head, rela, tail = triple[0], triple[1], triple[2]
        movie_head_triple_dict[(head, rela)].append(tail)
    for key, value in movie_head_triple_dict.items():
        if len(value) > 1:
            tail = ""
            for v in value:
                tail += v + ", "
                triples.remove([key[0], key[1], v])
            knowledge_str += "The " +  relation_head_tail_dict[key[1]][1] + " of " + relation_head_tail_dict[key[1]][0]+ " " + key[0] + " are: " + tail.strip(", ") + ". "            
    for triple in triples:
        head, rela, tail = triple[0], triple[1], triple[2]
        movie_tail_triple_dict[(tail, rela)].append(head)
    for key, value in movie_tail_triple_dict.items():
        if len(value) > 1 and key[1] not in ["has_imdb_rating", "has_imdb_votes"]:
            tail = ""
            for v in value:
                tail += v + ", "
                triples.remove([v, key[1], key[0]])
            knowledge_str += "The " +  relation_head_tail_dict[key[1]][0] + " which have " + relation_head_tail_dict[key[1]][1] + " " +  key[0] + " are: " + tail.strip(", ") + ". "  
    for triple in triples:
        knowledge_str += "The " + relation_head_tail_dict[triple[1]][1] + " of " + triple[0] + " is " + triple[2] + ". "
    return knowledge_str

def freebase_triple2text(triples):
    text = ""
    head_rela_dict = defaultdict(list)
    for triple in triples:
        (head_name, head_mid) = triple[0]
        relation = triple[1]
        (tail_name, tail_mid) = triple[2]
        head_rela_dict[(head_name, relation)].append(tail_name)
    for head_rela, tails in head_rela_dict.items():
        head, rela = head_rela
        t = "The " + rela.split(".")[-1] + " of " + head + " are: " 
        t += ", ".join(tails)
        t += ". "
        text += t
    return text