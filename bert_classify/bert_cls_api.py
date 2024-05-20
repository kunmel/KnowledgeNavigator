import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import config
import predict
import data_reader

class BertQuery(BaseModel):
    text:str
    
app = FastAPI()
args = config.ARGS
tokenizer = data_reader.load_tokenizer(config.ARGS)
args.model_dir = ""
args.task = "MetaQA-hop-predict" 
hop_classify_decoder = predict.load_classify_decoder(args)
hop_predict_model = predict.load_model(args)

@app.post("/bert_hop_predict/")
async def get_answer(bert_query:BertQuery):
    label_idx = predict.predict(hop_predict_model, args, bert_query.text, tokenizer)
    return {"answer": hop_classify_decoder[label_idx]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6009)

