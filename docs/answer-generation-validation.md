# Producción y validación de respuestas

Atlas genera respuestas en cuatro controles consecutivos:

1. **Umbral documental:** no se llama al modelo si la recuperación no produjo evidencia o si el mejor `rerank_score` está por debajo de `min_evidence_score`.
2. **Prompt restringido:** el modelo recibe únicamente la pregunta y el contexto RAG, con la obligación de citar cada afirmación mediante `[Fuente N]`.
3. **Validación determinista:** se comprueba que las fuentes citadas existan, que cada afirmación tenga cita, que las cifras estén presentes en la evidencia y que exista respaldo textual mínimo.
4. **Fallo cerrado:** una respuesta inválida puede repararse una vez. Si sigue sin respaldo, se reemplaza por un mensaje seguro.

## Proveedores

### Baseline extractivo

`extractive` copia la oración más relevante de los fragmentos recuperados. No es un LLM y no consume servicios externos. Sirve para pruebas locales, demostraciones reproducibles y diagnóstico del pipeline.

### OCI Generative AI

`oci` usa el SDK oficial de Oracle Cloud Infrastructure y la operación de chat de Generative AI. Admite:

- Autenticación mediante `~/.oci/config` para desarrollo.
- Instance Principals para el despliegue en una VM de OCI.
- Modelos bajo demanda indicados mediante su OCID.

Variables esperadas:

```text
OCI_COMPARTMENT_ID
OCI_GENAI_MODEL_ID
OCI_AUTH_MODE=config_file
OCI_CONFIG_FILE=~/.oci/config
OCI_CONFIG_PROFILE=DEFAULT
OCI_REGION
OCI_GENAI_ENDPOINT
```

`OCI_GENAI_ENDPOINT` es opcional; el SDK puede derivarlo desde la región.

## Estados de salida

- `answered`: respuesta generada y validada.
- `no_evidence`: no hubo evidencia suficiente y el modelo no fue llamado.
- `validation_failed`: la respuesta no superó la validación incluso después del intento de reparación.

## Limitaciones

La validación léxica y numérica reduce errores obvios, pero no constituye una prueba lógica completa. En una fase posterior se añadirá un conjunto de evaluación y, si aporta valor medible, un verificador basado en NLI o un segundo modelo.
