# -*- conding: utf-8 -*-
# @Time    : 2025/12/29  17:56
# @Author  : psi

import requests
import time


class VLLMResponse:
    def __init__(self, content: str):
        self.content = content


# 获取模型名称
def get_model_name(api_url):
    try:
        r = requests.get(f"{api_url}/v1/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["id"]
    except Exception as e:
        print("❌ 无法获取模型名称:", e)
        time.sleep(5)
        get_model_name(api_url)
        return None


class LLMService:

    def __init__(self):
        old_api_url = "http://127.0.0.1:8080"
        # if_one_by_one = False
        self.api_url = f"{old_api_url}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        # self.model_id = get_model_name(old_api_url)

    def predict(self, messages, enable_thinking: bool = False):
        """
        messages: list[dict] 
        enable_thinking: bool - 是否开启 thinking 模式（用于支持 CoT 的模型）
        
        """
        print(f"Calling LLM... (thinking={'ON' if enable_thinking else 'OFF'})")
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0,
            "top_p": 1.0,
            "n": 1,
            "logprobs": 1,  # 请求返回 token logprob
            "echo": False,  # 是否返回 prompt token
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=7200)
        response.raise_for_status()
        result = response.json()

        # 提取 assistant 的回复内容
        content = result["choices"][0]["message"]["content"]
        return VLLMResponse(content=content)

    # 启动 vllm 服务
    # nohup bash -c "CUDA_VISIBLE_DEVICES=0,1 vllm serve /data2/sre_rl/sre_out/new_model_save_vllm --port 8080 --tensor-parallel-size 2 --gpu-memory-utilization 0.7" > /data2/sre_rl/log/vllm.log 2>&1 &


if __name__ == '__main__':
    # 测试样例
    service = LLMService()
    
    # 构造测试消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 3? Answer with just the number."}
    ]
    
    print("Testing LLMService...")
    print(f"Model: {service.model_id}")
    print(f"API URL: {service.api_url}")
    print(f"Messages: {messages}")
    
    # 测试不开启 thinking
    print("\n" + "=" * 50)
    print("Test 1: enable_thinking=False")
    print("-" * 50)
    response = service.predict(messages, enable_thinking=False)
    print(f"Response content: {response.content}")
    
    # 测试开启 thinking
    print("\n" + "=" * 50)
    print("Test 2: enable_thinking=True")
    print("-" * 50)
    response = service.predict(messages, enable_thinking=True)
    print(f"Response content: {response.content}")

