from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from controllers.schemes.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from services import NLPService
from models import ResponseSignal
from tqdm.auto import tqdm
from stores.langgraph.graph import Graph  # Use this if you need to instantiate a graph
from stores.langgraph.scheme.state import AgentState
from stores.langgraph.graphFlow import GraphFlow
from stores.llm.templates.locales.en.symptom_sufficiency import symptom_sufficiency_prompt

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

# --- Graph setup ---
# You must create a GraphFlow and Graph instance with a real nlp_service.
# For FastAPI, you may want to create these per-request or as a dependency.
# Here is a simple example (replace with your actual nlp_service):
# from services.NLPService import NLPService
# nlp_service = NLPService(...)
# graph_flow = GraphFlow(AgentState(), nlp_service=nlp_service)
# graph = Graph(graph_flow)
#
# For now, we'll assume you have a function get_graph() that returns a ready-to-use graph instance.

def get_graph(request=None, project=None):
    """
    Returns a ready-to-use Graph instance with a real nlp_service and project context.
    If request and project are provided, uses them to build the correct context.
    """
    from services.NLPService import NLPService
    from stores.langgraph.graphFlow import GraphFlow
    from stores.langgraph.graph import Graph
    if request is None:
        raise RuntimeError("get_graph() requires the FastAPI request object for context.")
    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )
    graph_flow = GraphFlow(AgentState(), nlp_service=nlp_service)
    return Graph(graph_flow)

@nlp_router.post("/index/push/{project_id}", summary="Index project data into the vector database", description="Pushes all project chunks into the vector database for retrieval-augmented generation. Receives a project_id and a PushRequest (with do_reset flag). Returns the number of inserted items.")
async def index_project(request: Request, project_id: int, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
            }
        )
    
    nlp_controller = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    # create collection if not exists
    collection_name = nlp_controller.create_collection_name(project_id=project.project_id)

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )

    # setup batching
    total_chunks_count = await chunk_model.get_total_chunks_count(project_id=project.project_id)
    pbar = tqdm(total=total_chunks_count, desc="Vector Indexing", position=0)

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(project_id=project.project_id, page_no=page_no)
        if len(page_chunks):
            page_no += 1
        
        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids =  [ c.chunk_id for c in page_chunks ]
        idx += len(page_chunks)
        
        is_inserted = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            chunks_ids=chunks_ids
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                }
            )

        pbar.update(len(page_chunks))
        inserted_items_count += len(page_chunks)
        
    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}", summary="Get vector index info for a project", description="Retrieves metadata about the project's vector database collection, such as collection name and stats. Receives a project_id. Returns collection info.")
async def get_project_index_info(request: Request, project_id: int):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}", summary="Semantic search in project index", description="Performs a semantic search over the project's indexed data. Receives a project_id and a SearchRequest (with text and limit). Returns a list of matching results.")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    results = await nlp_controller.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value
                }
            )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [ result.dict()  for result in results ]
        }
    )

@nlp_router.post("/index/answer/{project_id}", summary="RAG answer from project index", description="Answers a question using retrieval-augmented generation over the project's indexed data. Receives a project_id and a SearchRequest (with text and limit). Returns the answer, full prompt, and chat history.")
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )

@nlp_router.post("/intent/classify", summary="Classify user intent", description="Classifies the intent of a user query. Receives a text string. Returns the intent label: SYMPTOM_TRIAGE (user describes symptoms or health complaints), MEDICAL_INFORMATION_REQUEST (user asks for medical info),OFF_TOPIC (query is unrelated to medical topics)")
async def classify_intent(request: Request, text: str):
    """Classify user intent."""
    try:
        state = AgentState(user_query=text, conversation_history="")
        graph = get_graph(request)  # <-- Use the correct graph instance
        result = await graph.classify_intent(state)
        return JSONResponse(
            content={
                "intent": result.get("intent", "OFF_TOPIC"),
                "confidence": result.get("confidence")
            }
        )
    except Exception as e:
        logger.error(f"Error in intent classification: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
    
@nlp_router.post("/symptoms/gather", summary="Extract symptoms from user input", description="Extracts and accumulates symptoms from a user query. Receives user_query (str) and optionally accumulated_symptoms (str). Returns the updated symptoms string.")
async def gather_symptoms(request: Request):
    data = await request.json()
    user_query = data.get("user_query", "")
    conversation_history = data.get("conversation_history", [])
    # Only accumulate user messages
    user_symptom_texts = [msg["content"] for msg in conversation_history if msg.get("role") == "user"]
    if user_query:
        user_symptom_texts.append(user_query)
    accumulated_symptoms = "\n".join(user_symptom_texts).strip()
    # ...call your symptom extraction logic here, passing accumulated_symptoms...
    # For demonstration, just return the accumulated_symptoms
    return {"accumulated_symptoms": accumulated_symptoms}

@nlp_router.post("/sufficiency/check", summary="Check symptom sufficiency", description="Checks if a symptom description is sufficiently detailed for analysis. Receives a text string. Returns a boolean is_sufficient and the raw answer.")
async def check_sufficiency(request: Request, text: str):
    """Check if the symptom description is sufficiently detailed for analysis."""
    try:
        result = await symptom_sufficiency_prompt.ainvoke({"accumulated_symptoms": text})
        logger.debug(f"Raw LLM response for sufficiency: {result!r}")
        # Handle result type robustly
        if hasattr(result, 'text'):
            answer = result.text.strip().upper()
        elif hasattr(result, 'to_string'):
            answer = result.to_string().strip().upper()
        else:
            answer = str(result).strip().upper()
        logger.debug(f"Parsed sufficiency answer: {answer}")
        is_sufficient = answer == "YES"
        return JSONResponse(
            content={
                "is_sufficient": is_sufficient,
                "raw_answer": answer
            }
        )
    except Exception as e:
        logger.error(f"Error in sufficiency check: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@nlp_router.post("/relevance/check", summary="Check triage relevance", description="Checks if a case is relevant for medical triage. Receives a text string. Returns a boolean is_relevant and confidence.")
async def check_relevance(request: Request, text: str):
    """Check if the case is relevant for triage."""
    try:
        # Pass the input as accumulated_symptoms for correct downstream logic
        state = AgentState(accumulated_symptoms=text)
        graph = get_graph(request)  # <-- Use the correct graph instance
        result = await graph.check_relevance(state)
        # Add a dummy confidence value for now
        return JSONResponse(
            content={
                "is_relevant": result.get("is_relevant", False),
                "confidence": 1.0
            }
        )
    except Exception as e:
        logger.error(f"Error in relevance check: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
    
@nlp_router.post("/info/process", summary="Process information request", description="Processes a general information request using the RAG pipeline. Receives query (str), docs (list), and web_results (list). Returns the processed response.")
async def process_information(request: Request, query: str, docs: list, web_results: list):
    """Process information requests."""
    try:
        state = AgentState(user_input=query)
        state.relevant_docs = docs
        state.web_results = web_results
        graph = get_graph(request)  # <-- Use the correct graph instance
        result = await graph._handle_information_request(state)
        return JSONResponse(
            content={"response": result.final_output}
        )
    except Exception as e:
        logger.error(f"Error in information processing: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@nlp_router.post("/analysis/perform", summary="Perform final analysis", description="Performs final analysis of symptoms and context. Receives symptoms (str) and docs (list). Returns analysis and relevant docs.")
async def perform_analysis(request: Request, symptoms: str, docs: list):
    """Perform final analysis."""
    try:
        state = AgentState(user_input=symptoms)
        state.relevant_docs = docs
        graph = get_graph(request)  # <-- Use the correct graph instance
        result = await graph._final_analysis(state)
        return JSONResponse(
            content={
                "analysis": result.final_analysis.analysis,
                "relevant_docs": result.final_analysis.relevant_docs
            }
        )
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@nlp_router.post("/explanation/evaluate", summary="Evaluate explanation quality", description="Evaluates the quality of a generated explanation. Receives explanation (str). Returns whether refinement is needed and confidence.")
async def evaluate_explanation(request: Request, explanation: str):
    """Evaluate explanation quality."""
    try:
        state = AgentState()
        state.final_analysis.analysis = explanation
        graph = get_graph(request)  # <-- Use the correct graph instance
        result = await graph._evaluate_explanation(state)
        return JSONResponse(
            content={
                "needs_refinement": result.explanation_evaluation["needs_refinement"],
                "confidence": result.explanation_evaluation["confidence"]
            }
        )
    except Exception as e:
        logger.error(f"Error in explanation evaluation: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@nlp_router.post("/explanation/refine", summary="Refine explanation if needed", description="Refines a generated explanation based on critique. Receives explanation (str) and critique (str). Returns the refined explanation.")
async def refine_explanation(request: Request, explanation: str, critique: str):
    """Refine explanation if needed."""
    try:
        state = AgentState()
        state.final_analysis.analysis = explanation
        state.explanation_evaluation = {"critique": critique}
        graph = get_graph(request)  # <-- Use the correct graph instance
        result = await graph._refine_explanation(state)
        return JSONResponse(
            content={"refined_explanation": result.final_analysis.analysis}
        )
    except Exception as e:
        logger.error(f"Error in explanation refinement: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@nlp_router.post("/output/prepare", summary="Prepare final output", description="Prepares the final output for the user. Receives analysis (str). Returns the final output string.")
async def prepare_output(request: Request, analysis: str):
    """Prepare final output."""
    try:
        # Use dict-style AgentState, not attribute access
        state = AgentState()
        state["final_explanation"] = analysis
        graph = get_graph(request)  # <-- Use the correct graph instance
        result = await graph.prepare_final_output(state)
        # result is expected to be a dict-like AgentState
        return JSONResponse(
            content={"output": result.get("final_output")}
        )
    except Exception as e:
        logger.error(f"Error in output preparation: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
    
@nlp_router.post("/triage/{project_id}", summary="Full triage pipeline with LangGraph", description="Runs the full medical triage pipeline using the LangGraph state machine. Receives a project_id and a SearchRequest (with text). Returns the final response and full state.")
async def triage_pipeline(request: Request, project_id: int):
    data = await request.json()
    user_query = data.get("text", "")
    conversation_history = data.get("conversation_history", [])
    # Only accumulate user messages
    user_symptom_texts = [msg["content"] for msg in conversation_history if msg.get("role") == "user"]
    if user_query:
        user_symptom_texts.append(user_query)
    accumulated_symptoms = "\n".join(user_symptom_texts).strip()
    # Build initial agent state
    state = AgentState(
        conversation_history=conversation_history,
        user_query=user_query,
        accumulated_symptoms=accumulated_symptoms,
        user_intent=None,
        is_relevant=None,
        loop_count=0,
        rag_context=None,
        initial_explanation=None,
        evaluator_critique=None,
        final_explanation=None,
        recommended_specialist=None,
        doctor_recommendations=None,
        no_doctors_found_specialist=None,
        final_response=None,
        uploaded_image_bytes=None,
        image_prompt_text=None,
    )
    # Run the graph pipeline
    graph = get_graph(request=request)
    result_state = await graph.run_full_triage(state)
    return {"final_response": result_state.get("final_response"), "state": result_state}




