
import os, ollama
from openai import OpenAI


class QueryLLM(object):
    def __init__(self, args):
        """
        GPT: gpt-3.5-turbo, gpt-4o-mini, gpt-4o

        Ollama: llama3.1:8b-instruct-fp16, llama3.1:70b-instruct-fp16
        """
        self.base_model = args['model']

        ## for GPT
        self.api_key = args['api'] if 'api' in args else None

        ## for Ollama
        self.host = args['host'] if 'host' in args else None
        

    def query_llama_model(self, prompt, system='', max_tokens=256, temperature=0):
        client = ollama.Client(
            host=self.host # f"http://59.77.7.22:11434"
        )
        response = client.generate(
            model=self.base_model,
            prompt=prompt,
            stream=False,
            # temperature=temperature,
        )
        # response = ollama.generate(
        #     model=self.args.model,
        #     prompt=prompt,
        #     stream=False,
        # )

        return response['response']

    def query_chatgpt_model(self, prompt, system='', max_tokens=256, temperature=0):
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        # "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"
        # api_key = "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"
        client = OpenAI(
            api_key=self.api_key,
            # base_url="https://api.xiaoai.plus/v1" # "https://api.chatanywhere.com.cn/v1"  # 国内转发需要
        )
        try:
            response = client.chat.completions.create(
                model=self.base_model,
                messages=[
                    {'role': 'system', "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens, n=1, stop=None, temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return False


    def forward(self, prompt, system='', max_tokens=256, temperature=0):
        if 'llama' in self.base_model:
            return self.query_llama_model(prompt, system=system, max_tokens=max_tokens, temperature=temperature)

        if 'gpt' in self.base_model:
            return self.query_chatgpt_model(prompt, system=system, max_tokens=max_tokens, temperature=temperature)