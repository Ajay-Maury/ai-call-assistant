import os
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_tavily import TavilySearch

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from aiVoiceAssistant.settings import TAVILY_API_KEY 


class LangChainAIAgent:
    """Enhanced AI Agent using LangChain with Tavily search tool"""
    
    def __init__(self):

        # self.llm = ChatGroq(
        #     model="llama-3.3-70b-versatile",  # You can use other supported Groq models
        #     temperature=0.7,
        #     api_key=GROQ_API_KEY
        # )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Tavily search tool
        self.tavily_search = TavilySearchResults(
            max_results=2,
            search_depth="basic",
            tavily_api_key=TAVILY_API_KEY
        )
        
        # Create custom tools list
        self.tools = [self.tavily_search]
        customer_name = "Aman"
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are **Manisha**, female voice, a warm, conversational voice agent from ** Aman Tech Innovations ** who always reply in hinglish in female context.

                Your objectives on every call are:
                1. Confirm the caller’s business context.(‣ Do they run/work for a business?)
                2. Introduce the Voice AI bot solution and ask permission to continue.
                3. Qualify the lead (outbound-call needs, volume, decision-maker, timeline).
                4. Move qualified leads forward; politely exit otherwise.

                ──────────────────
                ### Conversation Flow
                1. **Greeting & Context Check**  
                • If name known → “Hi {customer_name}, this is Manisha from Omen Tech Innovations!”  
                • Otherwise → “Hi there! This is Manisha from Omen Tech Innovations!”  
                • **Ask first:** “Do you run a business or work with one that handles customer calls?”

                2. **If ‘Yes’ → Brief Pitch + Permission**  
                • “Great! We’ve built a Voice AI bot that automates outbound calls and boosts customer engagement.”  
                • “Do you have a quick minute for me to share how it could help you?”

                **If ‘No’ →**  
                • “No worries—thanks for letting me know. Have a wonderful day!” (End call)

                3. **Lead-Qualification Questions**  
                • “Does your team make outbound calls for sales, reminders, or support?”  
                • “About how many outbound calls do you place—daily, monthly, or yearly?”  
                • Optional: “Who usually decides on call-automation tools in your company?”  
                • Optional: “Is there a budget or timeline for improving call operations?”

                4. **Next-Step Logic**  
                • **Qualified:** “Sounds like a great fit—shall we schedule a quick demo?”  
                • **Not qualified / no need:** “Thanks for the details. If things change, feel free to reach out. Have a great day!”

                ──────────────────
                ### Style Guidelines
                - Keep replies **1–2 sentences**, friendly and easy to understand aloud.
                - If the caller speaks in Hindi, respond in casual Hinglish (unless they request pure Hindi). Otherwise, match their language.
                - Use web-search only if asked for current info; summarize succinctly.
                - Always stay upbeat, respectful, and professional.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        
        # Store conversation memories per call
        self.memories: Dict[str, ConversationBufferMemory] = {}
        
        # Create agent executors per call (will be created on demand)
        self.agent_executors: Dict[str, AgentExecutor] = {}
    
    def get_memory(self, call_sid: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a specific call"""
        if call_sid not in self.memories:
            self.memories[call_sid] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.memories[call_sid]
    
    def get_agent_executor(self, call_sid: str) -> AgentExecutor:
        """Get or create agent executor for a specific call"""
        if call_sid not in self.agent_executors:
            memory = self.get_memory(call_sid)
            self.agent_executors[call_sid] = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
        return self.agent_executors[call_sid]
    
    def load_redis_context(self, call_sid: str, redis_context: List[Dict[str, str]]):
        """Load Redis context into LangChain memory"""
        if not redis_context:
            return
            
        memory = self.get_memory(call_sid)
        
        # Clear existing memory to avoid duplication
        memory.chat_memory.clear()
        
        # Load Redis context into LangChain memory
        print("\n\nbefore appending memory.chat_memory", memory.chat_memory)
        for message in redis_context:
            if message["role"] == "user":
                memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                memory.chat_memory.add_ai_message(message["content"])
        print("\n\nafter appending memory.chat_memory", memory.chat_memory)

    async def process_query(self, query: str, call_sid: str, redis_context: List[Dict[str, str]] = None) -> str:
        """Process a query using LangChain agent with conversation memory"""
        try:
            # Load Redis context if provided
            if redis_context:
                self.load_redis_context(call_sid, redis_context)
            
            # Get the agent executor for this call
            agent_executor = self.get_agent_executor(call_sid)
            
            # Process the query
            result = await agent_executor.ainvoke({"input": query})
            
            # Extract the response
            response = result.get("output", "I'm sorry, I couldn't process your request.")
            
            return response
            
        except Exception as e:
            print(f"Error in LangChain agent processing: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def clear_memory(self, call_sid: str):
        """Clear conversation memory for a specific call"""
        if call_sid in self.memories:
            del self.memories[call_sid]
        if call_sid in self.agent_executors:
            del self.agent_executors[call_sid]

    def get_conversation_history(self, call_sid: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific call in a readable format"""
        memory = self.get_memory(call_sid)
        messages = memory.chat_memory.messages
        
        history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    async def classify_user_input_type(self, user_text: str, call_sid: str) -> bool:
            """
            Classify the user input as 'engagement' or 'interrupt' using the LLM.
            """
            try:
                # Get the last 3 messages
                history = self.get_conversation_history(call_sid)
                recent_turns = history[-3:] if len(history) >= 3 else history
                recent_context = "\n".join(
                    f"{m['role']}: {m['content']}" for m in recent_turns
                )
    
                prompt = PromptTemplate.from_template("""
                    You are an assistant that classifies short user utterances during a phone call
                    as either 'engagement' (e.g. "haan", "hmm", "okay", "and") or 'interrupt' (e.g. "what did you say?", "I have a question").
                    also if the user is not saying anything meaningful or related to the conversation then consider it as 'engagement'.

                    Respond with boolean: true for "engagement" or false for "interrupt".

                    Recent conversation:
                    {recent_context}

                    User said: "{user_text}"
                """)

                output_parser = StrOutputParser()
                chain = prompt | self.llm | output_parser

                result = await chain.ainvoke({
                    "user_text": user_text.strip(),
                    "recent_context": recent_context
                })

                response = result.strip().lower()
                print(f"[Classifier] Classified '{user_text}' as engaged: {response}")
                return response == "true"

            except Exception as e:
                print(f"[Classifier] LLM error: {e}")
                return False
