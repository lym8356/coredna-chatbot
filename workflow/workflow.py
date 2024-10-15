import dotenv

from agent import ConciergeAgent
from workflow.events import AccountBalanceEvent, AuthenticateEvent, ConciergeEvent, InitializeEvent, OrchestratorEvent, StockLookupEvent, TransferMoneyEvent

dotenv.load_dotenv()

from llama_index.core.workflow import (
    step, 
    Context, 
    Workflow, 
    StartEvent, 
    StopEvent
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.utils.workflow import draw_all_possible_flows


class ConciergeWorkflow(Workflow):


    @step(pass_context=True)
    async def initialize(self, ctx: Context, ev: InitializeEvent) -> ConciergeEvent:

        return ConciergeEvent()
    

    @step(pass_context=True)
    async def concierge(self, ctx: Context, ev: ConciergeEvent | StartEvent) -> InitializeEvent | StopEvent | OrchestratorEvent:

        if ("index" not in ctx.data):
            return InitializeEvent()

        # initialize concierge if not already done
        if ("concierge" not in ctx.data):
            system_prompt = (f"""
                You are a helpful assistant that is helping a user navigate a financial system.
                Your job is to ask the user questions to figure out what they want to do, and give them the available things they can do.
                That includes

            """)

            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[],
                llm=ctx.data["llm"],
                allow_parallel_tool_calls=False,
                system_prompt=system_prompt
            )
            ctx.data["concierge"] = agent_worker.as_agent()

            concierge = ctx.data["concierge"]
            if ctx.data["overall_request"] is not None:
                print("There's an overall request in progress, it's ", ctx.data["overall_request"])
                last_request = ctx.data["overall_request"]
                ctx.data["overall_request"] = None
                return OrchestratorEvent(request=last_request)
            elif (ev.just_completed is not None):
                response = concierge.chat(f"FYI, the user has just completed the task: {ev.just_completed}")
            elif (ev.need_help):
                print("The previous process needs help with ", ev.request)
                return OrchestratorEvent(request=ev.request)
            else:
                # first time experience
                response = concierge.chat("Hello!")

            # print response
            print(Fore.MAGENTA + str(response) + Style.RESET_ALL)
            # here we need to fetch user input from client
            user_msg_str = input("> ").strip()

            return OrchestratorEvent(request=user_msg_str)
        
    @step(pass_context=True)
    async def orchestrator(self, ctx: Context, ev: OrchestratorEvent) -> ConciergeEvent  | StopEvent:

        print(f"Orchestrator received request: {ev.request}")

        def emit_stock_lookup() -> bool:
            """Call this if the user wants to look up a stock price."""      
            print("__emitted: stock lookup")      
            self.send_event(StockLookupEvent(request=ev.request))
            return True

        def emit_authenticate() -> bool:
            """Call this if the user wants to authenticate"""
            print("__emitted: authenticate")
            self.send_event(AuthenticateEvent(request=ev.request))
            return True

        def emit_account_balance() -> bool:
            """Call this if the user wants to check an account balance."""
            print("__emitted: account balance")
            self.send_event(AccountBalanceEvent(request=ev.request))
            return True

        def emit_transfer_money() -> bool:
            """Call this if the user wants to transfer money."""
            print("__emitted: transfer money")
            self.send_event(TransferMoneyEvent(request=ev.request))
            return True

        def emit_concierge() -> bool:
            """Call this if the user wants to do something else or you can't figure out what they want to do."""
            print("__emitted: concierge")
            self.send_event(ConciergeEvent(request=ev.request))
            return True

        def emit_stop() -> bool:
            """Call this if the user wants to stop or exit the system."""
            print("__emitted: stop")
            self.send_event(StopEvent())
            return True

        tools = [
            FunctionTool.from_defaults(fn=emit_stock_lookup),
            FunctionTool.from_defaults(fn=emit_authenticate),
            FunctionTool.from_defaults(fn=emit_account_balance),
            FunctionTool.from_defaults(fn=emit_transfer_money),
            FunctionTool.from_defaults(fn=emit_concierge),
            FunctionTool.from_defaults(fn=emit_stop)
        ]
        
        system_prompt = (f"""
            You are on orchestration agent.
            Your job is to decide which agent to run based on the current state of the user and what they've asked to do. 
            You run an agent by calling the appropriate tool for that agent.
            You do not need to call more than one tool.
            You do not need to figure out dependencies between agents; the agents will handle that themselves.
                            
            If you did not call any tools, return the string "FAILED" without quotes and nothing else.
        """)

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=ctx.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt
        )
        ctx.data["orchestrator"] = agent_worker.as_agent()        
        
        orchestrator = ctx.data["orchestrator"]
        response = str(orchestrator.chat(ev.request))

        if response == "FAILED":
            print("Orchestration agent failed to return a valid speaker; try again")
            return OrchestratorEvent(request=ev.request)
        

    @step(pass_context=True)
    def transfer_money(self, ctx: Context, ev: TransferMoneyEvent) -> AuthenticateEvent | AccountBalanceEvent | ConciergeEvent:

        if("transfer_money_agent" not in ctx.data):
            def transfer_money(from_account_id: str, to_account_id: str, amount: int) -> None:
                """Useful for transferring money between accounts."""
                print(f"Transferring {amount} from {from_account_id} account {to_account_id}")
                return f"Transferred {amount} to account {to_account_id}"
            
            def balance_sufficient(account_id: str, amount: int) -> bool:
                """Useful for checking if an account has enough money to transfer."""
                # todo: actually check they've selected the right account ID
                print("Checking if balance is sufficient")
                if ctx.data["user"]['account_balance'] >= amount:
                    return True
                
            def has_balance() -> bool:
                """Useful for checking if an account has a balance."""
                print("Checking if account has a balance")
                if ctx.data["user"]["account_balance"] is not None and ctx.data["user"]["account_balance"] > 0:
                    print("It does", ctx.data["user"]["account_balance"])
                    return True
                else:
                    return False
            
            def is_authenticated() -> bool:
                """Checks if the user has a session token."""
                print("Transfer money agent is checking if authenticated")
                if ctx.data["user"]["session_token"] is not None:
                    return True
                else:
                    return False
                
            def authenticate() -> None:
                """Call this if the user needs to authenticate."""
                print("Account balance agent is authenticating")
                ctx.data["redirecting"] = True
                ctx.data["overall_request"] = "Transfer money"
                self.send_event(AuthenticateEvent(request="Authenticate"))

            def check_balance() -> None:
                """Call this if the user needs to check their account balance."""
                print("Transfer money agent is checking balance")
                ctx.data["redirecting"] = True
                ctx.data["overall_request"] = "Transfer money"
                self.send_event(AccountBalanceEvent(request="Check balance"))
            
            system_prompt = (f"""
                You are a helpful assistant that transfers money between accounts.
                The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
                If they aren't authenticated, tell them to authenticate first.
                The user must also have looked up their account balance already, which you can check with the has_balance tool.
                If they haven't already, tell them to look up their account balance first.
                Once you have transferred the money, you can call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than transfer money, call the tool "done" to signal some other agent should help.
            """)

            ctx.data["transfer_money_agent"] = ConciergeAgent(
                name="Transfer Money Agent",
                parent=self,
                tools=[transfer_money, balance_sufficient, has_balance, is_authenticated, authenticate, check_balance],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=TransferMoneyEvent
            )

        return ctx.data["transfer_money_agent"].handle_event(ev)

