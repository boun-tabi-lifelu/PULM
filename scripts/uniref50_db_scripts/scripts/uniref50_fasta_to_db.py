import gzip
import os
import re
import sqlite3

# Header format (UniRef50 FASTA):
# >UniqueIdentifier ClusterName n=Members Tax=Taxon TaxID=TaxonID RepID=RepresentativeMember
# Older releases omit the TaxID= field, so it is optional here.
HEADER_RE = re.compile(
    r'^(?P<entry_id>\S+)'
    r'(?:\s+(?P<cluster_name>.*?))?'
    r'\s+n=(?P<member_count>\d+)'
    r'\s+Tax=(?P<taxon>.*?)'
    r'(?:\s+TaxID=(?P<taxon_id>\S*))?'
    r'\s+RepID=(?P<rep_id>\S+)\s*$'
)


def create_db(db_name="/cta/share/users/uniprot/uniref/uniref_2024_06/uniref50_representatives.db",
              table_name='uniref50_representatives'):
    """Initialize the SQLite database and create the table schema."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Bulk-load pragmas: big speed-up, safe because we can just rebuild on crash
    cursor.execute('PRAGMA journal_mode = OFF')
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA cache_size = -200000')  # ~200 MB page cache
    cursor.execute('PRAGMA temp_store = MEMORY')

    # Same schema as the XML version
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            entry_id TEXT PRIMARY KEY,
            cluster_name TEXT,
            member_count INTEGER,
            common_taxon_name TEXT,
            common_taxon_id TEXT,
            representative_member_id TEXT,
            sequence_length INTEGER,
            sequence TEXT
        )
    ''')
    conn.commit()
    return conn


def _open_maybe_gzip(path):
    """Open plain or .gz FASTA transparently as text."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def parse_header(header):
    """Parse a UniRef FASTA definition line (without the leading '>')."""
    m = HEADER_RE.match(header)
    if not m:
        # Fall back to at least capturing the ID so nothing is silently dropped
        entry_id = header.split(None, 1)[0] if header else None
        return entry_id, None, None, None, None, None

    cluster_name = m.group('cluster_name')
    if cluster_name is not None:
        cluster_name = cluster_name.strip() or None

    member_count = int(m.group('member_count'))

    taxon = m.group('taxon')
    taxon = taxon.strip() if taxon else None
    taxon = taxon or None

    taxon_id = m.group('taxon_id')
    taxon_id = taxon_id or None

    return (m.group('entry_id'), cluster_name, member_count,
            taxon, taxon_id, m.group('rep_id'))


def parse_and_insert(fasta_file, table_name, conn, batch_size=10000, report_every=1000000):
    """Stream the FASTA file and batch insert representative records into SQLite."""
    cursor = conn.cursor()

    insert_sql = f'''
        INSERT OR IGNORE INTO {table_name}
        (entry_id, cluster_name, member_count, common_taxon_name,
         common_taxon_id, representative_member_id, sequence_length, sequence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''

    records = []
    total = 0
    malformed = 0

    header = None
    seq_chunks = []

    def flush_record():
        """Turn the currently buffered header + sequence into a row."""
        nonlocal malformed
        entry_id, cluster_name, member_count, taxon, taxon_id, rep_id = parse_header(header)
        if member_count is None:
            malformed += 1
        sequence = ''.join(seq_chunks)
        records.append((
            entry_id, cluster_name, member_count,
            taxon, taxon_id,
            rep_id, len(sequence), sequence
        ))

    with _open_maybe_gzip(fasta_file) as fh:
        for line in fh:
            if line.startswith('>'):
                if header is not None:
                    flush_record()
                    total += 1

                    if len(records) >= batch_size:
                        cursor.executemany(insert_sql, records)
                        conn.commit()
                        records = []

                    if report_every and total % report_every == 0:
                        print(f"  ... {total:,} clusters processed", flush=True)

                header = line[1:].rstrip('\n').rstrip('\r')
                seq_chunks = []
            else:
                stripped = line.strip()
                if stripped:
                    seq_chunks.append(stripped)

        # Last entry in the file
        if header is not None:
            flush_record()
            total += 1

    if records:
        cursor.executemany(insert_sql, records)
        conn.commit()

    print(f"Inserted {total:,} clusters ({malformed:,} headers did not match the expected format).")
    return total


def build_indexes(conn, table_name):
    """Optional helper indexes for downstream lookups."""
    cursor = conn.cursor()
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_rep ON {table_name}(representative_member_id)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_taxid ON {table_name}(common_taxon_id)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_len ON {table_name}(sequence_length)')
    conn.commit()


if __name__ == '__main__':
    # File paths
    FASTA_FILE = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50.fasta'
    DB_FILE = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_representatives_fasta.db'

    # FASTA_FILE = 'sample2000.fasta'
    # DB_FILE = 'uniref50.db'

    TABLE_NAME = 'uniref50_representatives'

    print(f"Creating database {DB_FILE}...")
    db_conn = create_db(DB_FILE, TABLE_NAME)

    print(f"Parsing {FASTA_FILE} and loading to database. This will take some time...")
    parse_and_insert(FASTA_FILE, TABLE_NAME, db_conn)

    print("Building indexes...")
    build_indexes(db_conn, TABLE_NAME)

    db_conn.close()
    size_gb = os.path.getsize(DB_FILE) / (1024 ** 3)
    print(f"Database build successfully complete! ({size_gb:.2f} GB)")