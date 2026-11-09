import os, json
from openai import OpenAI

with open("data/train_sft.jsonl") as f:
    lines = list(filter(lambda n: len(n) > 1, f.read().split("\n")))

examples = [json.loads(l) for l in lines]

client = OpenAI(base_url="http://192.168.8.200:9931/v1", api_key="PLACEHOLDER")

conv_system_msg = {
    "role": "system",
    "content": "You are the LLM who came up with these reasoning traces.  Please summarize them to the best of your ability, preserving the minimal correct and necessary chain of reasoning.  It should not be necessary to keep specific prefix URLs, since they are provided elsewhere.  Don't include full draft queries, but do ensure the reasoning includes enough information to build the query from.  The reasoning content must not be blank."
}

with open("data/train_sft_minr_cont.jsonl", "w") as of:
    i = 1
    for x in examples[551:]:
        print(f'Example {i}')
        i += 1

        prev_rc = x["messages"][-1]["reasoning_content"]
        conv_msgs = [conv_system_msg, {"role": "user", "content": prev_rc}]
        resp = client.chat.completions.create(
            model='Qwen3.8-27B', messages=conv_msgs, temperature=0, max_tokens=1024,
            extra_body={"reasoning": {"effort": "medium"}})
        new_rc = resp.choices[0].message.content

        x["messages"][-1]["reasoning_content"] = new_rc

        if False:
            print("Prev. RC:")
            print("==============")
            print(prev_rc)
            print()

            print("New RC:")
            print("==============")
            print(new_rc)
            print()
            print()

        of.writelines([json.dumps(x) + '\n'])

    of.close()
