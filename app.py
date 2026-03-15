import logging
import asyncio
import operator
import tiktoken
from typing import Annotated, List, TypedDict, Optional, Literal, Union, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# --- CONFIGURATION LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SkunkWorks_v4_Final")

# --- SCHEMAS (IMMUTABLES) ---
class Evaluation(BaseModel):
    expert_name: str
    score: float = Field(..., ge=0, le=1)
    critique: str
    technical_flaws: List[str] = Field(default_factory=list)
    pivot_directive: Optional[str] = None

class Consensus(BaseModel):
    average_score: float
    status: Literal["continue", "synthesize", "hard_pivot", "emergency_stop"]
    unified_feedback: str
    iteration_count: int

class FinalBlueprint(BaseModel):
    titre: str
    architecture_summary: str
    key_mechanisms: List[str]
    equations_fondatrices: List[str]
    risk_mitigation: str
    verdict_final: Literal["Viable", "Théorique", "Irréalisable"]
    roadmap: List[str]

# --- STATE MANAGEMENT ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    iterations: int
    evaluations: Annotated[List[Evaluation], operator.add]
    current_consensus: Optional[Consensus]
    final_report: Optional[FinalBlueprint]
    metadata: Dict[str, Any]

# --- LLM FACTORY (SINGLETON PATTERN) ---
# Centralisation pour éviter les instanciations multiples
MODELS = {
    "gen": ChatOpenAI(model="gpt-4o", temperature=0.7),
    "critic": ChatOpenAI(model="gpt-4o-mini", temperature=0),
    "synth": ChatOpenAI(model="gpt-4o", temperature=0.1).with_structured_output(FinalBlueprint)
}

# --- NODES ---

async def generator_node(state: AgentState):
    iters = state.get("iterations", 0)
    consensus = state.get("current_consensus")
    
    # Stratégie d'Annealing (Réduction progressive de la créativité)
    temp = max(0.1, 0.8 - (iters * 0.2)) 
    
    # Gestion intelligente du contexte : On garde le system prompt et les derniers échanges
    trimmed = trim_messages(
        state["messages"],
        max_tokens=6000,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-4o"),
        start_on="human",
        include_system=True
    )

    # Construction dynamique du prompt de pilotage
    prompt_modifier = "Tu es l'Ingénieur en Chef. "
    if consensus:
        if consensus.status == "hard_pivot":
            # On ignore les messages précédents pour forcer une nouvelle direction
            prompt_modifier += f"RUPTURE REQUISE. L'approche précédente est obsolète. Change radicalement d'angle. Feedback : {consensus.unified_feedback}"
            trimmed = [trimmed[0], trimmed[-1]] # Garde l'instruction d'origine et la dernière critique
        else:
            prompt_modifier += f"AFFINAGE. Améliore la solution selon ces critiques : {consensus.unified_feedback}"

    full_prompt = [SystemMessage(content=prompt_modifier)] + trimmed
    response = await MODELS["gen"].bind(temperature=temp).ainvoke(full_prompt)
    
    return {"messages": [response], "iterations": iters + 1}

def expert_factory(role: str, name: str):
    """Factory optimisée avec gestion d'erreur granulaire."""
    async def _node(state: AgentState):
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if not last_ai:
            return {"evaluations": []}
        
        try:
            critic_chain = MODELS["critic"].with_structured_output(Evaluation)
            eval_res = await critic_chain.ainvoke([
                SystemMessage(content=f"Tu es l'Expert {role}. Analyse la rigueur, la sécurité et la faisabilité."),
                HumanMessage(content=f"Évalue la proposition technique suivante :\n\n{last_ai.content}")
            ])
            eval_res.expert_name = name
            return {"evaluations": [eval_res]}
        except Exception as e:
            logger.error(f"Critical Failure - Expert {name}: {e}")
            # Score neutre mais critique d'erreur pour ne pas bloquer le graphe
            return {"evaluations": [Evaluation(expert_name=name, score=0.5, critique="Erreur système d'analyse.")]}

    return _node

async def consensus_node(state: AgentState):
    # On récupère uniquement les évaluations de la DERNIÈRE itération (3 experts)
    last_evals = state["evaluations"][-3:]
    if not last_evals: return {}

    avg_score = sum(e.score for e in last_evals) / len(last_evals)
    
    # Logique de décision
    if avg_score > 0.88 or state["iterations"] >= 5:
        status = "synthesize"
    elif avg_score < 0.35:
        status = "hard_pivot"
    else:
        status = "continue"

    feedback = "\n".join([f"[{e.expert_name}] (Score: {e.score}): {e.critique}" for e in last_evals])
    
    return {
        "current_consensus": Consensus(
            average_score=avg_score,
            status=status,
            unified_feedback=feedback,
            iteration_count=state["iterations"]
        )
    }

async def synthesizer_node(state: AgentState):
    # On donne au synthétiseur le contexte complet : messages + toutes les critiques
    history = "\n---\n".join([m.content for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))])
    all_critiques = "\n".join([f"{e.expert_name}: {e.critique}" for e in state["evaluations"]])
    
    res = await MODELS["synth"].ainvoke([
        SystemMessage(content="Tu es le CTO. Compile les itérations de R&D en un Blueprint Final actionnable."),
        HumanMessage(content=f"HISTORIQUE :\n{history}\n\nCRITIQUES CUMULÉES :\n{all_critiques}")
    ])
    return {"final_report": res}

# --- ROUTING LOGIC ---
def router(state: AgentState):
    consensus = state.get("current_consensus")
    if not consensus: return "generator"
    
    return {
        "synthesize": "synthesizer",
        "hard_pivot": "generator",
        "continue": "generator"
    }.get(consensus.status, END)

# --- WORKFLOW CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Ajout des Nodes
workflow.add_node("generator", generator_node)
workflow.add_node("physics_expert", expert_factory("Physique/Hardware", "Alpha"))
workflow.add_node("security_expert", expert_factory("Cyber-sécurité", "Beta"))
workflow.add_node("logic_expert", expert_factory("Algorithmique", "Gamma"))
workflow.add_node("consensus", consensus_node)
workflow.add_node("synthesizer", synthesizer_node)

# Définition des liens
workflow.add_edge(START, "generator")

# Fan-out (Parallélisme)
workflow.add_edge("generator", "physics_expert")
workflow.add_edge("generator", "security_expert")
workflow.add_edge("generator", "logic_expert")

# Fan-in (Regroupement)
workflow.add_edge("physics_expert", "consensus")
workflow.add_edge("security_expert", "consensus")
workflow.add_edge("logic_expert", "consensus")

# Boucle conditionnelle
workflow.add_conditional_edges("consensus", router)
workflow.add_edge("synthesizer", END)

# Compilation avec persistance
app = workflow.compile(checkpointer=MemorySaver())
