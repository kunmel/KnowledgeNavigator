import model
import json
import config

with open(config.bert_input_path, "r") as input_f, open(config.bert_output_path, "w") as out_f:
    input_lines = input_f.readlines()
    for line in input_lines:
        line = line.strip()
        query_hop = int(model.bert_predict(line, "hop_predict"))
        l = json.dumps({"hop": query_hop}) +"\n"
        out_f.write(l)
        