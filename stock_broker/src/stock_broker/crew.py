from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_broker.tools.market_data_tool import MarketDataTool
from stock_broker.tools.technical_analysis_tool import TechnicalAnalysisTool
from stock_broker.tools.news_sentiment_tool import NewsSentimentTool
from stock_broker.tools.pattern_recognition_tool import PatternRecognitionTool
from stock_broker.tools.multi_timeframe_tool import MultiTimeframeTool
from stock_broker.tools.risk_management_tool import RiskManagementTool

@CrewBase
class StockBrokerCrew():
    """Advanced StockBroker crew with pattern recognition and multi-timeframe analysis"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def market_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['market_analyst'],
            tools=[MarketDataTool()],
            verbose=True
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_analyst'],
            tools=[TechnicalAnalysisTool()],
            verbose=True
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_analyst'],
            tools=[NewsSentimentTool()],
            verbose=True
        )

    @agent
    def pattern_recognition_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['pattern_recognition_analyst'],
            tools=[PatternRecognitionTool()],
            verbose=True
        )

    @agent
    def multi_timeframe_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['multi_timeframe_analyst'],
            tools=[MultiTimeframeTool()],
            verbose=True
        )

    @agent
    def risk_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_manager'],
            tools=[RiskManagementTool()],
            verbose=True
        )

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_manager'],
            tools=[],
            verbose=True
        )

    # Tasks
    @task
    def fetch_market_data(self) -> Task:
        return Task(config=self.tasks_config['fetch_market_data'])

    @task  
    def analyze_technical_indicators(self) -> Task:
        return Task(config=self.tasks_config['analyze_technical_indicators'])

    @task
    def analyze_news_sentiment(self) -> Task:
        return Task(config=self.tasks_config['analyze_news_sentiment'])

    @task
    def identify_chart_patterns(self) -> Task:
        return Task(config=self.tasks_config['identify_chart_patterns'])

    @task
    def multi_timeframe_analysis(self) -> Task:
        return Task(config=self.tasks_config['multi_timeframe_analysis'])

    @task
    def calculate_risk_parameters(self) -> Task:
        return Task(config=self.tasks_config['calculate_risk_parameters'])

    @task
    def generate_final_recommendation(self) -> Task:
        return Task(config=self.tasks_config['generate_final_recommendation'])

    @crew
    def crew(self) -> Crew:
        """Creates advanced StockBroker crew with 7 specialized agents"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
