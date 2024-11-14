from my_agent.agent import graph

def test_agent():
    # Test input
    input_message = {
        "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
    }
    
    # Execute
    try:
        response = graph.invoke(input_message)
        print("\nQuestion:", input_message["question"])
        print("\nResponse:", response)
        return True
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    test_agent()