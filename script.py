import json
from main import *


def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

if __name__ == "__main__":
    models = ["chatgpt"]
    tasks_filename ="task_names_test.json"

    opt = parse_opt()
    opt.num_episodes = 10
    opt.step = -1 # -1 means use the plan step generated by the llm
    opt.erci = 1
    opt.irci = 3
    opt.sgrounding = True
    opt.headless = True

    with open(tasks_filename) as f:
        task_names = json.load(f)["task_name"]
    
    for model in models :
        print("Using model : ", model)
        opt.llm = model
        for task_name in task_names :
            print("     Task addressed : ", task_name)
            opt.env = task_name     # switch task
            print(opt)
            #miiniwob(opt)

    