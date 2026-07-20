# Databricks notebook source
# MAGIC %md
# MAGIC # Load chunker output into Databricks AI Search
# MAGIC
# MAGIC (Databricks renamed Vector Search to **AI Search**; the SDK is now
# MAGIC `databricks-ai-search` and the old package is a deprecated shim.)
# MAGIC
# MAGIC Takes the Parquet tables produced by `intelligent-chunker export`
# MAGIC (`chunks.parquet`, `documents.parquet`, `glossary.parquet`) and:
# MAGIC
# MAGIC 1. Loads them into Delta tables (Unity Catalog).
# MAGIC 2. Computes **GTE-large v1.5** embeddings for `embedding_text`.
# MAGIC 3. Creates an **AI Search Delta Sync index** (self-managed embeddings).
# MAGIC 4. Shows a filtered + hybrid query.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - An AI Search endpoint (created below if missing).
# MAGIC - The GTE-large v1.5 HTTP endpoint URL + token (stored as a secret).
# MAGIC - The three Parquet files uploaded to a UC Volume.
# MAGIC - `embedding_text` is the column to embed; `text` is what you return.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-ai-search
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md ## Parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "spd")
dbutils.widgets.text("input_volume_path", "/Volumes/main/spd/exports/databricks_export")
dbutils.widgets.text("vs_endpoint", "spd-vector-search")
dbutils.widgets.text("gte_secret_scope", "spd")          # dbutils secret scope
dbutils.widgets.text("gte_url_key", "gte_endpoint_url")  # secret key: endpoint URL
dbutils.widgets.text("gte_token_key", "gte_token")       # secret key: bearer token

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
INPUT = dbutils.widgets.get("input_volume_path")
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint")

CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.chunks"
DOCS_TABLE = f"{CATALOG}.{SCHEMA}.documents"
GLOSSARY_TABLE = f"{CATALOG}.{SCHEMA}.glossary"
CHUNKS_INDEX = f"{CATALOG}.{SCHEMA}.chunks_index"
GLOSSARY_INDEX = f"{CATALOG}.{SCHEMA}.glossary_index"

EMBEDDING_DIM = 1024  # GTE-large v1.5

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Parquet → Delta (incremental)
# MAGIC
# MAGIC Change Data Feed must be on for a Delta Sync index to track updates.
# MAGIC
# MAGIC First load creates the table; later loads **MERGE on the content-hash
# MAGIC key** instead of overwriting. Unchanged rows keep their embeddings and
# MAGIC produce no CDF churn (a full overwrite would make Delta Sync re-index
# MAGIC every row on every run). Rows for a re-chunked document that no longer
# MAGIC exist in the new export are deleted — scoped to the doc_ids present in
# MAGIC the incoming batch so other documents in the table are untouched.

# COMMAND ----------

def load_parquet_to_delta(parquet_path: str, table_name: str, key: str = "id") -> None:
    df = spark.read.parquet(parquet_path)

    if not spark.catalog.tableExists(table_name):
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )
    else:
        df.createOrReplaceTempView("_incoming")
        cols = df.columns
        set_clause = ", ".join(f"t.{c} = s.{c}" for c in cols)
        insert_cols = ", ".join(cols)
        insert_vals = ", ".join(f"s.{c}" for c in cols)
        doc_ids = ", ".join(
            f"'{r.doc_id}'" for r in df.select("doc_id").distinct().collect()
        )
        has_embedding = "embedding" in spark.table(table_name).columns
        # A changed embedding_text with an unchanged key can't happen for
        # chunks (id hashes the text), but can for glossary (id hashes the
        # term only) — null the stale embedding so step 2 recomputes it.
        matched_clauses = (
            f"WHEN MATCHED AND t.embedding_text <> s.embedding_text "
            f"THEN UPDATE SET {set_clause}, t.embedding = NULL "
            f"WHEN MATCHED THEN UPDATE SET {set_clause} "
            if has_embedding and "embedding_text" in cols
            else f"WHEN MATCHED THEN UPDATE SET {set_clause} "
        )
        spark.sql(
            f"MERGE INTO {table_name} t USING _incoming s ON t.{key} = s.{key} "
            + matched_clauses
            + f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals}) "
            + (
                f"WHEN NOT MATCHED BY SOURCE AND t.doc_id IN ({doc_ids}) THEN DELETE"
                if doc_ids
                else ""
            )
        )

    spark.sql(
        f"ALTER TABLE {table_name} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    print(f"loaded {table_name}: {spark.table(table_name).count()} rows")


load_parquet_to_delta(f"{INPUT}/chunks.parquet", CHUNKS_TABLE)
load_parquet_to_delta(f"{INPUT}/documents.parquet", DOCS_TABLE, key="doc_id")
load_parquet_to_delta(f"{INPUT}/glossary.parquet", GLOSSARY_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Compute GTE embeddings for `embedding_text`
# MAGIC
# MAGIC Self-managed embeddings: we add an `embedding array<float>` column by
# MAGIC calling the external GTE HTTP endpoint, batched, via a pandas UDF.
# MAGIC
# MAGIC > If you instead host GTE on **Databricks Model Serving**, skip this
# MAGIC > cell and use the *Databricks-computed embeddings* index variant noted
# MAGIC > at the bottom — the index will embed `embedding_text` for you.

# COMMAND ----------

import time

import pandas as pd
import requests
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType

GTE_URL = dbutils.secrets.get(
    dbutils.widgets.get("gte_secret_scope"), dbutils.widgets.get("gte_url_key")
)
GTE_TOKEN = dbutils.secrets.get(
    dbutils.widgets.get("gte_secret_scope"), dbutils.widgets.get("gte_token_key")
)
EMBED_BATCH = 64
EMBED_RETRIES = 4  # on 429/5xx; one transient blip must not fail the whole job


def gte_embed(texts):
    """POST a batch of texts to the GTE endpoint -> list[list[float]].

    Retries 429/5xx with exponential backoff (honoring Retry-After) so one
    transient endpoint error can't fail the entire Spark write.

    TODO: adjust the request/response shape to match your endpoint contract.
    The shape below (``{"inputs": [...]}`` -> ``{"embeddings": [[...]]}``) is a
    common one; change the keys if yours differs.
    """
    delay = 2.0
    for attempt in range(EMBED_RETRIES + 1):
        resp = requests.post(
            GTE_URL,
            headers={"Authorization": f"Bearer {GTE_TOKEN}"},
            json={"inputs": list(texts)},
            timeout=120,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == EMBED_RETRIES:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else delay)
            delay *= 2
            continue
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
            raise ValueError(
                f"Endpoint returned {len(embeddings[0])}-dim vectors, "
                f"expected EMBEDDING_DIM={EMBEDDING_DIM} — wrong model behind "
                "the endpoint?"
            )
        return embeddings


@pandas_udf(ArrayType(FloatType()))
def embed_udf(texts: pd.Series) -> pd.Series:
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(gte_embed(texts.iloc[i : i + EMBED_BATCH].tolist()))
    return pd.Series(out)


def add_embeddings(table_name: str) -> None:
    """Embed only rows that need it (new/changed); first run embeds all."""
    df = spark.table(table_name)
    if "embedding" not in df.columns:
        embedded = df.withColumn("embedding", embed_udf("embedding_text"))
        (
            embedded.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )
        spark.sql(
            f"ALTER TABLE {table_name} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
        return

    pending = df.where("embedding IS NULL")
    n = pending.count()
    if n == 0:
        print(f"{table_name}: embeddings up to date")
        return
    print(f"{table_name}: embedding {n} new/changed rows")
    pending.withColumn(
        "embedding", embed_udf("embedding_text")
    ).createOrReplaceTempView("_embedded")
    spark.sql(
        f"MERGE INTO {table_name} t USING _embedded s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET t.embedding = s.embedding"
    )


add_embeddings(CHUNKS_TABLE)
add_embeddings(GLOSSARY_TABLE)   # so "what does X mean" queries hit definitions
display(spark.table(CHUNKS_TABLE).select("id", "section_title", "embedding").limit(3))

# COMMAND ----------

# MAGIC %md ## 3. Create the AI Search endpoint + indexes

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

vsc = AISearchClient(disable_notice=True)

existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"creating endpoint {VS_ENDPOINT} ...")
vsc.wait_for_endpoint(VS_ENDPOINT, verbose=True)

# COMMAND ----------

def create_self_managed_index(source_table: str, index_name: str) -> None:
    try:
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",          # or "CONTINUOUS" for live sync
            primary_key="id",
            embedding_dimension=EMBEDDING_DIM,
            embedding_vector_column="embedding",
        )
        print(f"creating index {index_name} ...")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"{index_name} exists; syncing")
            vsc.get_index(VS_ENDPOINT, index_name).sync()
        else:
            raise


create_self_managed_index(CHUNKS_TABLE, CHUNKS_INDEX)
create_self_managed_index(GLOSSARY_TABLE, GLOSSARY_INDEX)

vsc.get_index(VS_ENDPOINT, CHUNKS_INDEX).wait_until_ready(verbose=True)
vsc.get_index(VS_ENDPOINT, GLOSSARY_INDEX).wait_until_ready(verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Query
# MAGIC
# MAGIC Embed the query with the **same** GTE model, retrieve top-k, and return
# MAGIC the chunk plus its section/page provenance. Filters scope the search;
# MAGIC `query_type="HYBRID"` blends vector + keyword (good for exact terms like
# MAGIC plan names, dollar limits, IRS section numbers).

# COMMAND ----------

def search(question, k=5, filters=None, hybrid=True):
    qvec = gte_embed([question])[0]
    index = vsc.get_index(VS_ENDPOINT, CHUNKS_INDEX)
    res = index.similarity_search(
        query_vector=qvec,
        query_text=question if hybrid else None,   # hybrid needs the text too
        query_type="HYBRID" if hybrid else "ANN",
        columns=["id", "text", "section_title", "section_type",
                 "page_start", "page_end", "plan_name", "keywords"],
        filters=filters or {},
        num_results=k,
    )
    cols = [c["name"] for c in res["manifest"]["columns"]]
    return [dict(zip(cols, row)) for row in res["result"]["data_array"]]


for hit in search(
    "When do I become fully vested in employer contributions?",
    k=5,
    # Example filters (omit for all): scope to one plan / section type / pages.
    # filters={"section_type": "vesting"},
    # filters={"plan_name": "AHS Retirement Partnership 401(k) Plan"},
    # filters={"page_start >=": 5},
):
    print(f"[{hit['section_title']} p{hit['page_start']}-{hit['page_end']}] "
          f"{hit['text'][:160]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes / next steps
# MAGIC
# MAGIC - **Re-running the chunker**: `export` writes stable content-hash `id`s,
# MAGIC   so reloading Parquet + a `TRIGGERED` `.sync()` updates changed rows.
# MAGIC   Use `CONTINUOUS` pipeline_type if you want near-real-time sync.
# MAGIC - **Databricks-computed embeddings (alternative to cell 2)**: the
# MAGIC   Foundation Model API serves the same model as `databricks-gte-large-en`
# MAGIC   (1024-dim, pay-per-token). Drop the embedding UDF and create the index
# MAGIC   with `embedding_source_column="embedding_text"` +
# MAGIC   `embedding_model_endpoint_name="databricks-gte-large-en"` instead of
# MAGIC   `embedding_vector_column`/`embedding_dimension`. The index embeds for
# MAGIC   you (queries too, so `search()` no longer needs to embed the question).
# MAGIC - **Reranking**: for precision, retrieve a wider `k` (e.g. 20) and rerank
# MAGIC   before sending to the LLM. AI Search now has this built in: pass
# MAGIC   `reranker=DatabricksReranker(columns_to_rerank=["text"])` to
# MAGIC   `similarity_search` (`from databricks.ai_search.reranker import
# MAGIC   DatabricksReranker`).
# MAGIC - **Cross-references**: after retrieval, optionally pull the sections named
# MAGIC   in a hit's `cross_references` from `documents.sections_json` to make the
# MAGIC   answer more comprehensive.
# MAGIC - **Glossary index**: query `GLOSSARY_INDEX` the same way for definitional
# MAGIC   questions, and use it to expand acronyms in the user's query first.
