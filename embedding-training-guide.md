# Fine-tuning all-MiniLM-L6-v2 para o ViBo

## Porquê fine-tunar?

O modelo base `all-MiniLM-L6-v2` foi treinado em texto genérico da internet.
O ViBo tem vocabulário específico: notas pessoais, tarefas, wikilinks, frontmatter YAML, terminologia de produtividade em português/inglês.

Fine-tunar significa que o modelo aprende que:
- `[[Projecto ViBo]]` e `"ViBo project"` são semanticamente próximos
- `#high priority` e `"urgente"` são próximos
- `due: 2025-06-01` e `"deadline de junho"` são próximos

Resultado: pesquisa semântica e SRI routing muito mais precisos para o contexto do utilizador.

---

## O que precisas

### Hardware mínimo
- GPU com 4GB VRAM (ou Google Colab gratuito)
- 8GB RAM
- 2GB espaço em disco

### Software
```bash
pip install sentence-transformers datasets torch
```

---

## Formato dos dados de treino

O modelo usa **contrastive learning** — pares de frases (similar / diferente).

### Estrutura do ficheiro JSONL

```jsonl
{"anchor": "criar nota sobre reunião", "positive": "nova nota reunião de equipa", "negative": "encriptar ficheiro vault"}
{"anchor": "mover card para done", "positive": "tarefa concluída muda coluna", "negative": "pesquisar nota por tag"}
{"anchor": "[[Projecto ViBo]]", "positive": "ViBo application notes", "negative": "receita de culinária"}
{"anchor": "due: 2025-06-01", "positive": "deadline primeiro de junho", "negative": "nota diária de hoje"}
{"anchor": "calendário amanhã", "positive": "evento google calendar tomorrow", "negative": "encriptar nota privada"}
{"anchor": "#high priority", "positive": "urgente importante agora", "negative": "someday talvez mais tarde"}
```

### Tipos de pares que deves criar

| Categoria | Âncora | Positivo | Negativo |
|---|---|---|---|
| Intenção de nota | "nova nota sobre X" | "criar ficheiro md X" | "apagar tarefa" |
| Wikilinks | "[[Nome da Nota]]" | "referência a Nome da Nota" | "tag #categoria" |
| Kanban intent | "mover para in progress" | "tarefa em curso" | "pesquisar notas antigas" |
| Calendar intent | "agendar reunião" | "criar evento calendar" | "nota diária" |
| Prioridade | "#high due amanhã" | "urgente deadline tomorrow" | "backlog someday" |
| Vault/encrypt | "bloquear nota" | "encriptar conteúdo privado" | "partilhar nota" |
| Daily note | "nota de hoje" | "daily note today" | "nota arquivo 2023" |
| SRI routing | "preciso de ajuda com X" | "pergunta sobre X" | acção não relacionada |

---

## Script de treino

```python
# train_embeddings.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

# 1. Carrega modelo base
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Carrega dados
def load_training_data(path: str):
    examples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            examples.append(InputExample(
                texts=[item["anchor"], item["positive"], item["negative"]]
            ))
    return examples

train_examples = load_training_data("vibo_training.jsonl")

# 3. DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 4. Loss — TripletLoss é ideal para este formato
train_loss = losses.TripletLoss(model=model)

# 5. Treino
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="vibo-minilm-finetuned",
    show_progress_bar=True,
)

print("Modelo guardado em: vibo-minilm-finetuned/")
```

---

## Quantidade de dados recomendada

| Fase | Pares de treino | Resultado esperado |
|---|---|---|
| Mínimo viável | 500 pares | Melhorias notáveis em routing ViBo |
| Bom | 2.000 pares | Pesquisa semântica sólida |
| Óptimo | 5.000+ pares | Routing quase perfeito |

Para começar, 500 pares bem escolhidos são suficientes.

---

## Como gerar dados de treino automaticamente

Podes usar o Claude ou o LFM2 para gerar pares a partir das notas reais do utilizador:

```python
# generate_pairs.py — usa Claude API para gerar pares de treino
import anthropic, json, os

client = anthropic.Anthropic()

def generate_pairs_from_note(note_content: str, n: int = 10) -> list:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""
Dado este conteúdo de nota, gera {n} pares de treino para fine-tuning de embeddings.
Cada par deve ter: anchor (query de pesquisa), positive (texto semanticamente similar), negative (texto não relacionado).
Responde apenas em JSON array.

Nota:
{note_content}
"""
        }]
    )
    return json.loads(response.content[0].text)
```

---

## Exportar para usar no ViBo

Após treino, o modelo fica em `vibo-minilm-finetuned/`.
Para usar no Rust via `candle` ou via sidecar Python/ONNX:

```bash
# Exportar para ONNX (mais fácil de integrar no Rust/Android)
pip install optimum

optimum-cli export onnx \
  --model vibo-minilm-finetuned \
  --task feature-extraction \
  vibo-minilm-onnx/
```

O ficheiro `vibo-minilm-onnx/model.onnx` é o que integras no ViBo para inferência on-device.

---

## Integração com storage.rs

Após exportar o ONNX, o fluxo no ViBo:

```
Texto da nota
    → all-MiniLM fine-tuned (ONNX, on-device)
    → Vec<f32> com 384 dimensões
    → storage_store_embedding()
    → sqlite-vec

Query do utilizador
    → all-MiniLM fine-tuned (mesma inferência)
    → Vec<f32> com 384 dimensões
    → storage_semantic_search()  (cosine similarity sub-20ms)
```

---

## Notas importantes

1. **Privacidade total** — treino acontece na cloud (Colab), mas o modelo resultante corre on-device. As notas do utilizador nunca saem do dispositivo.
2. **Re-treino** — podes re-treinar periodicamente com novos dados se o vocabulário do utilizador mudar muito.
3. **Dimensões fixas** — o modelo fine-tunado mantém 384 dimensões. O schema do sqlite-vec não muda.
4. **Língua** — o MiniLM base já suporta português razoavelmente. Com fine-tuning em pares PT/EN melhora significativamente.
