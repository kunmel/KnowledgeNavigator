import config
import prompt
import model
import utils 
import requests
from urllib.parse import quote

def retrieve_with_relation_quality_freebase(all_queries, core_entities, query_hop):
    knowledge_triple_list = []
    current_entities = []
    relation_path = []
    already_read_entity_set = set()
    current_entities = core_entities
    for iter in range(query_hop):
        iter_relation_list = []
        iter_tail_entitymid = []
        for idx, (entity, mid) in enumerate(current_entities):
            if (entity, mid) in already_read_entity_set:
                continue
            else:
                already_read_entity_set.add((entity, mid))
            relation_dict, relation_count, top_relations, relations_join_str = freebase_get_relation_with_entity(entity, mid)
            if relation_count > config.FREEBASE_ENTITY_RELATION_THRESHOLD:
                for i in range(4):
                    relation_sort_list = freebase_llm_relation_scoring(all_queries, entity, relations_join_str)
                    if len(relation_sort_list) >= 1 :
                        best_relations = relation_sort_list[:config.FREEBASE_ENTITY_RELATION_THRESHOLD]
                        break
            else:
                best_relations = top_relations
            iter_relation_list.append(best_relations)
            triples_list = []
            tail_entities_list = []
            for best_relation in best_relations:   
                triples, tail_entities = freebase_get_triple_with_relation(entity, mid, relation_dict[best_relation])
                triples_list += triples
                tail_entities_list += tail_entities
            for triple in triples_list:
                if triple not in knowledge_triple_list:
                    knowledge_triple_list.append(triple) 
            for (entity_name, mid) in tail_entities_list:
                if (entity_name, mid) not in iter_tail_entitymid:
                    iter_tail_entitymid.append((entity_name, mid)) 
        current_entities = iter_tail_entitymid
        relation_path.append(iter_relation_list)
    return utils.freebase_triple2text(knowledge_triple_list), relation_path


def freebase_get_triple_with_relation(entity, mid, best_relation):
    triples = []
    tail_entities = []
    sparql_template = """
    PREFIX ns:<http://rdf.freebase.com/ns/>
    SELECT DISTINCT ns:{mid} ns:{best_relation} ?e2 ?e2_name
    WHERE {{
    ns:{mid} ns:{best_relation} ?e2
    OPTIONAL {{?e2 ns:type.object.name ?e2_name FILTER(lang(?e2_name)='en')}}
    }}
    """
    sparql_template_2 = """
    PREFIX ns:<http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?e2 ns:{best_relation} ns:{mid} ?e2_name
    WHERE {{
    ?e2 ns:{best_relation} ns:{mid}
    OPTIONAL {{?e2 ns:type.object.name ?e2_name FILTER(lang(?e2_name)='en')}}
    }}
    """
    sparql = sparql_template.format(mid = mid, best_relation = best_relation)
    encoded_query = quote(sparql)
    full_url = f"{config.freebase_url}?query={encoded_query}"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(full_url, headers=headers)
    if response.status_code == 200:
        results = response.json()
        for binding in results['results']['bindings']:
            tail_mid = binding['e2']['value'].split("/")[-1]
            if "e2_name" in binding:
                tail_entities.append((binding['e2_name']['value'], tail_mid))
                triples.append([(entity, mid), best_relation, (binding['e2_name']['value'], tail_mid)])
            else:
                tail_entities.append((tail_mid, tail_mid))
                triples.append([(entity, mid), best_relation, (tail_mid, tail_mid)])
    else:
        print(f"SPARQL query failed with status code {response.status_code}")
        print(response.text)
    sparql = sparql_template_2.format(mid = mid, best_relation = best_relation)
    encoded_query = quote(sparql)
    full_url = f"{config.freebase_url}?query={encoded_query}"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(full_url, headers=headers)
    if response.status_code == 200:
        results = response.json()
        for binding in results['results']['bindings']:
            tail_mid = binding['e2']['value'].split("/")[-1]
            if "e2_name" in binding:
                tail_entities.append((binding['e2_name']['value'], tail_mid))
                triples.append([(binding['e2_name']['value'], tail_mid), best_relation, (entity, mid)])
            else:
                tail_entities.append((tail_mid, tail_mid))
                triples.append([(tail_mid, tail_mid), best_relation, (entity, mid)])
    else:
        print(f"SPARQL query failed with status code {response.status_code}")
        print(response.text)
    return triples, tail_entities

def freebase_get_relation_with_entity(mid):
    relations = []
    sparql_template = """
    PREFIX ns:<http://rdf.freebase.com/ns/>
    SELECT DISTINCT ns:{mid} ?r
    WHERE {{
    ns:{mid} ?r ?e2
    }}
    """
    sparql_template_2 = """
    PREFIX ns:<http://rdf.freebase.com/ns/>
    SELECT DISTINCT ns:{mid} ?r
    WHERE {{
    ?e2 ?r  ns:{mid}
    }}
    """
    sparql = sparql_template.format(mid = mid)
    encoded_query = quote(sparql)
    full_url = f"{config.freebase_url}?query={encoded_query}"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(full_url, headers=headers)
    if response.status_code == 200:
        results = response.json()
        for binding in results['results']['bindings']:
            relation = binding['r']['value'].split("/")
            if relation[-2] == 'ns':
                relations.append(relation[-1])
    else:
        print(f"SPARQL query failed with status code {response.status_code}")
        print(response.text)
    sparql = sparql_template_2.format(mid = mid)
    encoded_query = quote(sparql)
    full_url = f"{config.freebase_url}?query={encoded_query}"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(full_url, headers=headers)
    if response.status_code == 200:
        results = response.json()
        for binding in results['results']['bindings']:
            relation = binding['r']['value'].split("/")
            if relation[-2] == 'ns':
                relations.append(relation[-1])
    else:
        print(f"SPARQL query failed with status code {response.status_code}")
        print(response.text)
    relation_dict = {}
    relation_join_str = ""
    for relation in relations:
        if "type.object" in relation:
            continue
        last_relation = ".".join([relation.split(".")[-2], relation.split(".")[-1]])
        relation_join_str += last_relation + "; "
        relation_dict[last_relation] = relation
    return relation_dict, len(relations), relations[:config.FREEBASE_ENTITY_RELATION_THRESHOLD], relation_join_str


def freebase_llm_relation_scoring(all_queries, entity,relations):
    relation_vote = {}
    promtpts= []
    for idx, query in enumerate(all_queries):
        promtpts.append(prompt.get_relation_retrieve_prompt_fewshot(query, relations, config.LLM_RELATION_CHOOSE_NUMBER, entity))
    llm_relation_retrieves = model.chat_llama2_multi(promtpts)
    for idx, llm_relation_retrieve in enumerate(llm_relation_retrieves):    
        flag, best_relations = utils.decode_relation_retrieve(llm_relation_retrieve, relations)
        if flag:
            score = 1.0 if idx == 0 else 0.51
            for best_relation in best_relations:
                if best_relation in relation_vote:
                    relation_vote[best_relation] += score
                else:
                    relation_vote[best_relation] = score
    sorted_relations = dict(sorted(relation_vote.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_relations.keys())



