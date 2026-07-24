# Capa de recuperación RAG

Esta etapa transforma una pregunta en evidencia documental lista para el modelo generativo.

## Flujo

1. La pregunta se vectoriza con el mismo proveedor y modelo usados durante la indexación.
2. Chroma recupera un conjunto amplio de candidatos por similitud vectorial.
3. Los filtros de metadatos se aplican en la consulta vectorial.
4. Los candidatos pasan por una segunda etapa de reclasificación.
5. Se eliminan duplicados y se limita la cantidad de fragmentos por archivo.
6. Los fragmentos finales se convierten en un contexto con citas y presupuesto máximo.

## Filtros disponibles

- `category`
- `file_type`
- `source_file`
- `status`
- `version`
- `owner`

La aplicación consultará por defecto documentos con `status=vigente`.

## Rerankers

### Híbrido

Es el modo predeterminado. Combina el score vectorial con cobertura de términos, coincidencia de frases y números. No descarga modelos adicionales y funciona como segunda etapa reproducible para el MVP.

### FastEmbed cross-encoder

Es opcional y se activa con `--reranker fastembed`. Descarga un modelo adicional y evalúa directamente cada pareja pregunta-fragmento. Debe evaluarse en español antes de establecerlo como opción de producción.

### Sin reranking

`--reranker none` conserva el orden vectorial y sirve como línea base para comparar la calidad.

## Protección del contexto

Los documentos se tratan como evidencia no confiable, no como instrucciones. El contexto indica al LLM que ignore órdenes incrustadas dentro de los documentos. Esta medida reduce el riesgo de prompt injection documental, aunque no sustituye las pruebas de seguridad de la aplicación completa.

## Ejecución

```bash
python scripts/retrieve_context.py \
  "¿Cuántos días de vacaciones recibe un colaborador nuevo?" \
  --category "Recursos Humanos" \
  --candidate-k 4 \
  --final-k 3
```

El comando imprime:

- candidatos recuperados;
- scores vectoriales y de reranking;
- fuentes y ubicaciones;
- contexto completo para el LLM;
- salida JSON para integrar posteriormente con la interfaz.

## Parámetros iniciales

| Parámetro | Valor |
|---|---:|
| Candidatos vectoriales | 12 |
| Fragmentos finales | 4 |
| Máximo por archivo | 2 |
| Presupuesto de contexto | 6000 caracteres |
| Estado documental | vigente |

Estos valores son iniciales y deberán ajustarse mediante un conjunto de preguntas de evaluación.
