"""Neo4j loader. Schema:

  (:Person {key})            canonical normalized name
  (:Place {key})
  (:Year {value})
  (:Record {id, collection, title, url})
  (Record)-[:MENTIONS {source}]->(Person|Place)
  (Record)-[:DATED]->(Year)
  (Person)-[:SAME_AS {score, reason}]-(Person)   cross-collection identity links
"""
from __future__ import annotations

from collections.abc import Iterable

from rcgraph.entities import Entities
from rcgraph.link import Link

SCHEMA = [
    "CREATE CONSTRAINT person_key IF NOT EXISTS FOR (p:Person) REQUIRE p.key IS UNIQUE",
    "CREATE CONSTRAINT place_key IF NOT EXISTS FOR (p:Place) REQUIRE p.key IS UNIQUE",
    "CREATE CONSTRAINT year_value IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE",
    "CREATE CONSTRAINT record_id IF NOT EXISTS FOR (r:Record) REQUIRE r.id IS UNIQUE",
    "CREATE FULLTEXT INDEX person_ft IF NOT EXISTS FOR (p:Person) ON EACH [p.key]",
]

UPSERT_RECORD = """
UNWIND $rows AS row
MERGE (r:Record {id: row.id})
SET r.collection = row.collection, r.title = row.title, r.url = row.url
WITH r, row
UNWIND row.people AS p
  MERGE (person:Person {key: p})
  MERGE (r)-[:MENTIONS {source: row.source}]->(person)
WITH r, row
UNWIND row.places AS pl
  MERGE (place:Place {key: pl})
  MERGE (r)-[:MENTIONS {source: row.source}]->(place)
WITH r, row
UNWIND row.years AS y
  MERGE (year:Year {value: y})
  MERGE (r)-[:DATED]->(year)
"""

UPSERT_LINK = """
UNWIND $rows AS row
MATCH (a:Person {key: row.a}), (b:Person {key: row.b})
MERGE (a)-[s:SAME_AS]-(b)
SET s.score = row.score, s.reason = row.reason
"""


class GraphLoader:
    def __init__(self, uri: str, user: str, password: str, batch_size: int = 500):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size

    def close(self):
        self.driver.close()

    def init_schema(self):
        with self.driver.session() as s:
            for stmt in SCHEMA:
                s.run(stmt)

    def _batched(self, query: str, rows: Iterable[dict]):
        buf = []
        with self.driver.session() as s:
            for r in rows:
                buf.append(r)
                if len(buf) >= self.batch_size:
                    s.run(query, rows=buf)
                    buf = []
            if buf:
                s.run(query, rows=buf)

    def load_records(self, records: Iterable[tuple[dict, Entities]]):
        def rows():
            for meta, ents in records:
                yield {**meta, "people": sorted(ents.people), "places": sorted(ents.places),
                       "years": sorted(ents.years)}
        self._batched(UPSERT_RECORD, rows())

    def load_links(self, links: Iterable[Link]):
        self._batched(UPSERT_LINK, ({"a": lk.a.name, "b": lk.b.name, "score": lk.score,
                                     "reason": lk.reason} for lk in links))
