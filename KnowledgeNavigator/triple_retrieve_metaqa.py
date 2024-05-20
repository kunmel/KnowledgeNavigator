import config
import prompt
import model
import utils 


def retrieve_with_relation_quality(all_queries, core_entities, query_hop, head_entity_labels):
    knowledge_triple_list = []
    current_entities = []
    relation_path = []
    already_read_entity_set = set()
    for idx, entity in enumerate(core_entities):
        current_entities.append((entity, head_entity_labels[idx]))
    for i in range(query_hop):
        iter_relation_list = []
        iter_tail_entity_label = []
        for (entity, label) in current_entities:
            if (entity, label) in already_read_entity_set:
                continue
            else:
                already_read_entity_set.add((entity, label))
            relations = ""
            relations_list = entity2relation(entity, label)
            relations = "; ".join(relations_list)
            if len(relations_list) > config.METAQA_ENTITY_RELATION_THRESHOLD:
                for i in range(3):
                    relation_sort_list = llm_relation_scoring(all_queries, entity, relations, label)
                    if len(relation_sort_list) != 0 :
                        best_relation = relation_sort_list[:config.METAQA_ENTITY_RELATION_THRESHOLD]
                        break
            else:
                best_relation = relations_list
            iter_relation_list.append(best_relation)
            triples_list = []
            tail_entities_list = []
            tail_labels_list = []
            for relation in best_relation:
                triples, tail_entities, tail_labels = get_triple_with_relation(entity, label, relation)
                triples_list += triples
                tail_entities_list += tail_entities
                tail_labels_list += tail_labels   
            for triple in triples_list:
                if triple not in knowledge_triple_list:
                    knowledge_triple_list.append(triple) 
            for idx, entity in enumerate(tail_entities_list):
                if (entity, tail_labels_list[idx]) not in iter_tail_entity_label:
                    iter_tail_entity_label.append((entity, tail_labels_list[idx])) 
        current_entities = iter_tail_entity_label
        relation_path.append(iter_relation_list)
    return utils.triple2text(knowledge_triple_list), relation_path

def llm_relation_scoring(all_queries, entity, relations):
    relation_vote = {}
    prompts = []
    for idx, query in enumerate(all_queries):
        relation_retrieve_prompt = prompt.get_relation_retrieve_prompt_fewshot(query, relations, config.LLM_RELATION_CHOOSE_NUMBER,entity)
        prompts.append(relation_retrieve_prompt)
    llm_relation_retrieves = model.chat_llama2_multi(prompts)
    for idx, llm_answer in enumerate(llm_relation_retrieves):
        success, best_relations = utils.decode_relation_retrieve(llm_answer)
        if not success:
            return []
        score = 1.0 if idx == 0 else 0.51
        for best_relation in best_relations:
            if best_relation in relation_vote:
                relation_vote[best_relation] += score
            else:
                relation_vote[best_relation] = score
    sorted_relations = dict(sorted(relation_vote.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_relations.keys())
        
def get_triple_with_relation(entity, label, relation):
    triples = []
    tail_entities = []
    tail_labels = []
    cql = f"""MATCH (m:{label})-[r:`{relation}`]->(n) WHERE m.name = "{entity}" RETURN m.name as head, type(r) as rela, n.name as tail, labels(n) as tail_label"""
    dataList = config.graph.run(cql).data()
    for data in dataList:
        triples.append([data['head'], data['rela'], data['tail']])
        tail_entities.append(data['tail'])
        tail_labels.append(data['tail_label'][0])
    cql = f"""MATCH (m:{label})<-[r:`{relation}`]-(n) WHERE m.name = "{entity}" RETURN m.name as tail, type(r) as rela, n.name as head, labels(n) as head_label"""
    dataList = config.graph.run(cql).data()
    for data in dataList:
        triples.append([data['head'], data['rela'], data['tail']])
        tail_entities.append(data['head'])
        tail_labels.append(data['head_label'][0])
    return triples, tail_entities, tail_labels      

def entity2relation(entity, label):
    relations = []
    cql = f"""MATCH (m:{label})-[r]->(n) WHERE m.name="{entity}" RETURN DISTINCT type(r) as r"""
    dataList = config.graph.run(cql).data()
    for relation in dataList:
        relations.append((relation["r"]))
    cql = f"""MATCH (n)-[r]->(m:{label}) WHERE m.name="{entity}" RETURN DISTINCT type(r) as r"""
    dataList = config.graph.run(cql).data()
    for relation in dataList:
        relations.append(relation["r"])
    return relations