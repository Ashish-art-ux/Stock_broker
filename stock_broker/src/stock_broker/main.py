import sys
from stock_broker.crew import StockBrokerCrew

def run():
    """
    Run the crew
    """
    inputs = {
        'symbol': 'AAPL',
        'interval': '1min'
    }

    StockBrokerCrew().crew().kickoff(inputs = inputs)

def train():
    """
    Train the crew for a given number of iteration.
    """
    inputs = {
        'symbol': 'AAPL',
        'interval': '1min'

    }
    try:
        StockBrokerCrew().crew().train(n_iterations = int(sys.argv[1]),inputs = inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        StockBrokerCrew().crew().replay(task_id=sys.argv[9])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

if __name__ == "__main__":
    run()
