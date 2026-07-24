# Interfaz, despliegue y mantenimiento de Atlas

## Objetivo

Esta etapa convierte el pipeline RAG en una aplicación web accesible mediante Streamlit y prepara su ejecución persistente con Docker y Oracle Cloud Infrastructure.

## Interfaz

La aplicación principal es `streamlit_app.py` e incluye:

- Aviso visible de que Atlas es un agente de IA.
- Historial de conversación durante la sesión.
- Filtro por categoría documental.
- Respuestas con estado, confianza y proveedor.
- Fuentes verificadas en un panel expandible.
- Feedback positivo o negativo por respuesta.
- Métricas básicas de calidad en la barra lateral.
- Manejo amistoso de errores e índice no inicializado.

El historial de conversación vive en `st.session_state`; no se conserva después de cerrar la sesión del navegador.

## Ejecución local

```bash
cp .env.example .env
python scripts/bootstrap_demo_index.py
streamlit run streamlit_app.py
```

La aplicación queda disponible en `http://localhost:8501`.

## Ejecución con Docker

```bash
cp .env.example .env
docker compose up --build
```

El contenedor:

1. Comprueba si existe un índice.
2. Crea el índice de demostración cuando es necesario.
3. Inicia Streamlit en `0.0.0.0:8501`.
4. Expone un health check en `/_stcore/health`.
5. Conserva datos y caché en volúmenes Docker.

## Proveedores de respuesta

### Baseline extractivo

```env
ATLAS_LLM_PROVIDER=extractive
```

Es gratuito, local y determinista. Sirve para validar el producto completo antes de habilitar costos y permisos cloud.

### OCI Generative AI

```env
ATLAS_LLM_PROVIDER=oci
OCI_COMPARTMENT_ID=...
OCI_GENAI_MODEL_ID=...
OCI_AUTH_MODE=instance_principal
OCI_REGION=...
```

En una instancia de OCI se recomienda `instance_principal` para evitar copiar claves privadas dentro del contenedor o repositorio.

## Eventos y privacidad

Los eventos se almacenan localmente en:

`data/feedback/events.jsonl`

Por defecto no se almacena el texto de la pregunta. Se registra:

- hash de la pregunta;
- longitud;
- estado de la respuesta;
- proveedor y modelo;
- confianza documental;
- latencia;
- archivos citados;
- feedback positivo o negativo.

Para guardar preguntas completas durante una evaluación controlada:

```env
ATLAS_LOG_QUERY_TEXT=true
```

Debe activarse solo cuando exista una política de privacidad y los usuarios sepan qué información se conserva.

## Reporte de métricas

```bash
python scripts/metrics_report.py
```

El reporte incluye:

- total de preguntas;
- respuestas producidas;
- preguntas sin evidencia;
- fallos de validación;
- tasa de feedback negativo;
- latencia promedio;
- confianza documental promedio.

## Actualización de documentos

Coloca documentos aprobados en:

`knowledge-base/documents/`

Actualiza el inventario:

`knowledge-base/document-inventory.csv`

Después ejecuta:

```bash
python scripts/refresh_knowledge_base.py
```

El proceso:

1. Extrae y limpia todos los documentos.
2. Cancela la publicación si algún archivo falla, salvo que se use `--allow-partial`.
3. Genera embeddings en una colección de staging.
4. Comprueba que todos los chunks estén indexados.
5. Publica atómicamente el nuevo manifiesto.
6. Conserva la colección anterior para rollback manual.

La aplicación debe reiniciarse después de publicar un índice nuevo porque el runtime se mantiene en caché.

## Rutina de mantenimiento sugerida

### Semanal

- Ejecutar `refresh_knowledge_base.py`.
- Revisar documentos con errores.
- Revisar preguntas sin respuesta y feedback negativo.

### Mensual

- Confirmar versiones y responsables del inventario.
- Retirar documentos obsoletos.
- Añadir preguntas recurrentes al conjunto de evaluación.
- Comparar la calidad antes de cambiar modelo o prompt.

### Antes de cambiar el modelo

- Ejecutar todas las pruebas.
- Ejecutar el conjunto de evaluación.
- Comparar precisión, tasa sin respuesta, latencia y costo.
- Mantener un mecanismo de rollback.

## Archivos de despliegue

- `Dockerfile`: imagen de producción.
- `docker-compose.yml`: ejecución persistente.
- `.streamlit/config.toml`: configuración del servidor.
- `deploy/atlas.service`: inicio automático mediante systemd.
- `deploy/atlas-refresh.service`: actualización del índice.
- `deploy/atlas-refresh.timer`: ejecución semanal.

## Seguridad mínima

- No publicar `.env`, claves privadas ni archivos de configuración de OCI.
- Usar Instance Principals en OCI Compute.
- Abrir únicamente los puertos necesarios.
- Limitar SSH a la dirección IP del administrador cuando sea posible.
- Colocar HTTPS mediante proxy inverso antes de utilizar datos empresariales reales.
- No habilitar registro de preguntas completas sin una política de privacidad.
