from my_agent.agent import graph

def test_agent():
    input_message = {
        "messages": [("user", "What are the current trends in apartment sales?")]
    }
    
    try:
        response = graph.invoke(input_message)
        print("\nQuestion:", input_message["messages"][0][1])
        print("\nResponse:", response)
        return True
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    test_agent()