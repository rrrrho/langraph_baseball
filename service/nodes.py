from service.agents import batting_agent, pitching_agent, team_agent, games_ws_agent, games_playoffs_agent, games_regular_agent
from model.state import State
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

class Node_Service:
    llm = None

    def __init__(self, model):
        self.llm = model
        self.graph = self.create_workflow()

    def ask(self, state: State):
        result = self.graph.invoke(state)
        return result

    def route_question(self, state: State) -> State:
        """Use an LLM to intelligently route the question to the appropriate agent node."""
        
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing assistant for a baseball data analysis system. 
            Your job is to analyze the user's question and determine which dataset would be most appropriate to answer it.
            
            Available datasets and their nodes:
            - batting_node: Player batting statistics (batting average, hits, home runs, RBIs, OPS, slugging)
            - pitching_node: Pitcher statistics (ERA, WHIP, strikeouts, saves, innings pitched)
            - team_node: Team-level statistics (wins, losses, win percentage, run differential, standings)
            - ws_games_node: World Series game results (scores, winners, game-by-game results)
            - playoff_games_node: Playoff games excluding World Series (Wild Card, Division Series, Championship Series, ALCS, NLCS, ALDS, NLDS)
            - regular_games_node: Regular season game results (scores, dates, matchups from April through September)
            
            Respond with ONLY the node name, nothing else. For example: "batting_node" or "pitching_node"
            
            If the question could apply to multiple datasets, choose the most specific one.
            If you're unsure, default to "ws_games_node"."""),
            ("user", "{question}")
        ])

        routing_chain = routing_prompt | self.llm

        try:
            response = routing_chain.invoke({"question": state["question"]})
            data_source = response.content.strip()
            
            valid_nodes = [
                "batting_node", "pitching_node", "team_node",
                "ws_games_node", "playoff_games_node", "regular_games_node"
            ]
            
            if data_source not in valid_nodes:
                print(f"LLM returned invalid node '{data_source}', defaulting to ws_games_node")
                data_source = "ws_games_node"
            else:
                print(f"LLM routing to: {data_source}")
                
        except Exception as e:
            print(f"Error in LLM routing: {e}, defaulting to ws_games_node")
            data_source = "ws_games_node"
        
        return {
            "question": state["question"],
            "data_source": data_source,
            "answer": "",
        }

    def batting_node(self, state: State) -> State:
        """Query the batting statistics agent."""
        try:
            result = batting_agent(self.llm).invoke({"input": state["question"]})
            answer = result.get("output") or result.get("output_text") or str(result)
        except Exception as e:
            answer = f"Error querying batting stats: {str(e)[:200]}"
            print(f"Error occurred: {answer}")
        
        return {
            "question": state["question"],
            "data_source": state["data_source"],
            "answer": answer
        }


    def pitching_node(self, state: State) -> State:
        """Query the pitching statistics agent."""
        try:
            result = pitching_agent(self.llm).invoke({"input": state["question"]})
            answer = result.get("output") or result.get("output_text") or str(result)
        except Exception as e:
            answer = f"Error querying pitching stats: {str(e)[:200]}"
            print(f"Error occurred: {answer}")
        
        return {
            "question": state["question"],
            "data_source": state["data_source"],
            "answer": answer
        }


    def team_node(self, state: State) -> State:
        """Query the team statistics agent."""
        try:
            result = team_agent(self.llm).invoke({"input": state["question"]})
            answer = result.get("output") or result.get("output_text") or str(result)
        except Exception as e:
            answer = f"Error querying team stats: {str(e)[:200]}"
            print(f"Error occurred: {answer}")
        
        return {
            "question": state["question"],
            "data_source": state["data_source"],
            "answer": answer
        }


    def ws_games_node(self, state: State) -> State:
        """Query the World Series games agent."""
        try:
            result = games_ws_agent(self.llm).invoke({"input": state["question"]})
            answer = result.get("output") or result.get("output_text") or str(result)
        except Exception as e:
            answer = f"Error querying World Series games: {str(e)[:200]}"
            print(f"Error occurred: {answer}")
        
        return {
            "question": state["question"],
            "data_source": state["data_source"],
            "answer": answer
        }


    def playoff_games_node(self, state: State) -> State:
        """Query the playoff games agent."""
        try:
            result = games_playoffs_agent(self.llm).invoke({"input": state["question"]})
            answer = result.get("output") or result.get("output_text") or str(result)
        except Exception as e:
            answer = f"Error querying playoff games: {str(e)[:200]}"
            print(f"Error occurred: {answer}")
        
        return {
            "question": state["question"],
            "data_source": state["data_source"],
            "answer": answer
        }


    def regular_games_node(self, state: State) -> State:
        """Query the regular season games agent."""
        try:
            result = games_regular_agent(self.llm).invoke({"input": state["question"]})
            answer = result.get("output") or result.get("output_text") or str(result)
        except Exception as e:
            answer = f"Error querying regular season games: {str(e)[:200]}"
            print(f"Error occurred: {answer}")
        
        return {
            "question": state["question"],
            "data_source": state["data_source"],
            "answer": answer
        }
    
    def create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("route", self.route_question)
        workflow.add_node("batting_node", self.batting_node)
        workflow.add_node("pitching_node", self.pitching_node)
        workflow.add_node("team_node", self.team_node)
        workflow.add_node("ws_games_node", self.ws_games_node)
        workflow.add_node("playoff_games_node", self.playoff_games_node)
        workflow.add_node("regular_games_node", self.regular_games_node)
        workflow.set_entry_point("route")

        workflow.add_conditional_edges(
            "route",                
            lambda state: state["data_source"],
            {                         
                "batting_node": "batting_node",
                "pitching_node": "pitching_node",
                "team_node": "team_node",
                "ws_games_node": "ws_games_node",
                "playoff_games_node": "playoff_games_node",
                "regular_games_node": "regular_games_node"
            }
        )
        workflow.add_edge("batting_node", END)
        workflow.add_edge("pitching_node", END)
        workflow.add_edge("team_node", END)
        workflow.add_edge("ws_games_node", END)
        workflow.add_edge("playoff_games_node", END)
        workflow.add_edge("regular_games_node", END)

        return workflow.compile()