from __future__ import annotations

import csv
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas.app import AtlasRuntime, RuntimeSettings  # noqa: E402
from atlas.monitoring import EventStore, summarize_events  # noqa: E402

st.set_page_config(
    page_title="Atlas | Asistente corporativo",
    page_icon="🧭",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_runtime() -> AtlasRuntime:
    return AtlasRuntime(RuntimeSettings.from_environment(PROJECT_ROOT))


@st.cache_resource(show_spinner=False)
def load_event_store() -> EventStore:
    path = os.getenv("ATLAS_EVENTS_PATH", "data/feedback/events.jsonl")
    event_path = Path(path)
    if not event_path.is_absolute():
        event_path = PROJECT_ROOT / event_path
    return EventStore(event_path)


@st.cache_data(show_spinner=False)
def load_categories(inventory_path: str) -> list[str]:
    path = Path(inventory_path)
    if not path.is_file():
        return []
    categories: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            category = (row.get("category") or "").strip()
            if category:
                categories.add(category)
    return sorted(categories)


def initialize_session() -> None:
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("feedback", {})


def reference_dict(reference: Any) -> dict[str, Any]:
    return {
        "citation": reference.citation,
        "source_file": reference.source_file,
        "location": reference.location,
        "category": reference.category,
        "version": reference.version,
    }


def save_feedback(message: dict[str, Any], rating: str) -> None:
    event_id = message.get("answer_event_id")
    if not event_id or event_id in st.session_state.feedback:
        return
    load_event_store().record_feedback(
        session_id=st.session_state.session_id,
        answer_event_id=event_id,
        rating=rating,
    )
    st.session_state.feedback[event_id] = rating


def render_sources(references: list[dict[str, Any]]) -> None:
    if not references:
        return
    with st.expander(f"Fuentes verificadas ({len(references)})"):
        for reference in references:
            st.markdown(f"**{reference['citation']}**")
            details = [
                f"Archivo: `{reference['source_file']}`",
                f"Ubicación: {reference['location'] or 'no especificada'}",
            ]
            if reference.get("category"):
                details.append(f"Categoría: {reference['category']}")
            if reference.get("version"):
                details.append(f"Versión: {reference['version']}")
            st.caption(" · ".join(details))


def render_feedback(message: dict[str, Any], index: int) -> None:
    event_id = message.get("answer_event_id")
    if not event_id:
        return
    selected = st.session_state.feedback.get(event_id)
    left, right, note = st.columns([1, 1, 5])
    with left:
        if st.button(
            "👍",
            key=f"positive-{index}-{event_id}",
            help="La respuesta fue útil",
            disabled=selected is not None,
        ):
            save_feedback(message, "positive")
            st.rerun()
    with right:
        if st.button(
            "👎",
            key=f"negative-{index}-{event_id}",
            help="La respuesta necesita mejorar",
            disabled=selected is not None,
        ):
            save_feedback(message, "negative")
            st.rerun()
    with note:
        if selected:
            st.caption("Feedback registrado. Gracias.")
        else:
            st.caption("¿Esta respuesta fue útil?")


def render_message(message: dict[str, Any], index: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("references", []))
            status = message.get("status", "unknown")
            evidence = float(message.get("evidence_score", 0.0))
            provider = message.get("provider", "unknown")
            st.caption(
                f"Estado: `{status}` · Confianza documental: `{evidence:.3f}` · "
                f"Proveedor: `{provider}`"
            )
            for warning in message.get("warnings", []):
                st.warning(warning)
            render_feedback(message, index)


initialize_session()

st.title("🧭 Atlas")
st.caption(
    "Agente corporativo de inteligencia artificial de NovaCommerce. "
    "Sus respuestas se basan en documentos internos ficticios y muestran las fuentes utilizadas."
)
st.info(
    "Estás conversando con un sistema de IA, no con una persona. "
    "Verifica las fuentes antes de tomar decisiones sensibles."
)

try:
    runtime = load_runtime()
    health = runtime.health()
except Exception as error:  # noqa: BLE001 - friendly UI boundary
    st.error("Atlas todavía no puede iniciar porque el índice vectorial no está disponible.")
    st.code("python scripts/bootstrap_demo_index.py", language="bash")
    with st.expander("Detalle técnico"):
        st.exception(error)
    st.stop()

with st.sidebar:
    st.header("Configuración")
    if health.ready:
        st.success("Índice listo")
    else:
        st.error("Índice vacío")
    st.metric("Fragmentos indexados", health.indexed_chunks)
    st.caption(f"Colección: `{health.collection_name}`")
    st.caption(f"Generación: `{health.llm_provider}`")

    categories = load_categories(str(runtime.settings.inventory_path))
    selected_category = st.selectbox(
        "Filtrar por categoría",
        options=["Todas"] + categories,
        index=0,
    )

    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

    st.divider()
    st.subheader("Calidad de la demostración")
    metrics = summarize_events(load_event_store().iter_events())
    st.metric("Preguntas", metrics.total_questions)
    st.metric("Sin respuesta", f"{metrics.unanswered_rate:.0%}")
    st.metric("Feedback negativo", f"{metrics.negative_feedback_rate:.0%}")
    st.caption(
        "El historial del chat vive solo durante esta sesión. Las métricas locales no se suben a GitHub."
    )

for message_index, stored_message in enumerate(st.session_state.messages):
    render_message(stored_message, message_index)

question = st.chat_input("Escribe una pregunta sobre los documentos de NovaCommerce")
if question:
    clean_question = question.strip()
    if clean_question:
        user_message = {"role": "user", "content": clean_question}
        st.session_state.messages.append(user_message)
        render_message(user_message, len(st.session_state.messages) - 1)

        with st.status("Buscando evidencia y validando la respuesta...", expanded=False) as status:
            started = time.perf_counter()
            try:
                result = runtime.answer(
                    clean_question,
                    category=None if selected_category == "Todas" else selected_category,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                answer_event = load_event_store().record_answer(
                    session_id=st.session_state.session_id,
                    query=clean_question,
                    result=result,
                    latency_ms=latency_ms,
                    include_query_text=runtime.settings.log_query_text,
                )
                assistant_message = {
                    "role": "assistant",
                    "content": result.answer,
                    "status": result.status,
                    "provider": result.provider,
                    "model": result.model,
                    "evidence_score": result.evidence_score,
                    "references": [reference_dict(item) for item in result.references],
                    "warnings": result.warnings,
                    "answer_event_id": answer_event["event_id"],
                }
                st.session_state.messages.append(assistant_message)
                status.update(label="Respuesta validada", state="complete")
            except Exception as error:  # noqa: BLE001
                status.update(label="No se pudo producir la respuesta", state="error")
                assistant_message = {
                    "role": "assistant",
                    "content": (
                        "Ocurrió un error al consultar la base documental. "
                        "Revisa la configuración o intenta nuevamente."
                    ),
                    "status": "error",
                    "provider": runtime.settings.llm_provider,
                    "evidence_score": 0.0,
                    "references": [],
                    "warnings": [str(error)],
                }
                st.session_state.messages.append(assistant_message)
        render_message(assistant_message, len(st.session_state.messages) - 1)
