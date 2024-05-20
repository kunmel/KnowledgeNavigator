import model 
import utils
import prompt
import config
import json

with open(config.data_question_path, "r") as q_f, open(config.data_similar_question_path, "a") as similar_f:
    q_lines = q_f.readlines()
    for line in q_lines:
        line = line.strip()
        similar_prompt = prompt.get_similar_question_prompt(config.SIMILAR_QUERY_GENERATION_COUNT, line)    
        for i in range(config.MAX_SIMILAR_QUERY_GENERATION_ROUND):
            similar_queries = model.chat_llama2([similar_prompt])
            success, similar_queries = utils.decode_similar_queries(similar_queries)
            if success:
                break
        similar_f.write(json.dumps({"similar_questions": similar_queries}) + "\n")
        similar_f.flush()


