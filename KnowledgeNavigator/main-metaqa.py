import model 
import prompt
import config
import triple_retrieve_metaqa as triple_retrieve
import json


def get_answer(question, similar_q, hop_predict, core_entity, entity_label):
    all_queries = [question] + similar_q
    knowledge, relation_path = triple_retrieve.retrieve_with_relation_quality(all_queries, [core_entity], hop_predict, [entity_label])
    answer_prompt= prompt.get_answer_prompt_fewshot(question, knowledge)
    llm_answer = model.chat_llama2([answer_prompt])
    return llm_answer, knowledge, relation_path


with open(config.KN_output_path, "w") as output_file, open(config.data_question_path, "r") as q_file, open(config.data_gold_answer_path, "r") as a_file, open(config.data_similar_question_path, "r") as similar_file, open(config.predict_path, "r") as predict_file:
    q_lines = q_file.readlines()
    similar_qlines = similar_file.readlines()
    predict_lines = predict_file.readlines()
    a_lines = a_file.readlines()
    for idx, q in enumerate(q_lines):
        question = json.loads(q)["query"]
        core_entity = json.loads(q)["entity_name"]
        entity_label = json.loads(q)["entity_label"]
        similar_q = json.loads(similar_qlines[idx])["similar_questions"]
        hop_predict = json.loads(predict_lines[idx])["hop"]
        gold_answer = a_lines[idx].strip()
        llm_answer, knowledge, relation_path = get_answer(question, similar_q, hop_predict, core_entity, entity_label)
        result = {}
        result["correct answer"] = gold_answer
        result["llm answer"] = llm_answer
        result["knowledge"] = knowledge
        result["relation path"]  = relation_path
        output_file.write(json.dumps(result) + "\n")
