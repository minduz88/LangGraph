# test_agent.py
from my_agent.agent import graph

def test_agent():
    # 테스트 구성
    config = {"model_name": "anthropic"}  # 또는 "openai"
    
    # 테스트 입력
    input_message = {
        "messages": [{
            "role": "user",
            "content": "What is 2+2?"
        }]
    }
    
    # 실행
    try:
        response = graph.invoke(input_message, config=config)
        print("Response:", response)
        return True
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    test_agent()