import os
import sqlite3

# lxml is 2-3x faster than the stdlib parser; fall back gracefully if absent.
try:
    from lxml import etree as ET
    HAVE_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    HAVE_LXML = False

NS = '{http://uniprot.org/uniref}'
ENTRY_TAG = NS + 'entry'
NAME_TAG = NS + 'name'
PROP_TAG = NS + 'property'
REP_TAG = NS + 'representativeMember'
DBREF_TAG = NS + 'dbReference'
SEQ_TAG = NS + 'sequence'


def create_db(db_name, table_name='uniref50_representatives'):
    """Initialize the SQLite database and create the table schema."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Bulk-load pragmas. These are the single biggest win on a shared/network
    # filesystem: no rollback journal and no fsync on every commit.
    cursor.execute('PRAGMA journal_mode = OFF')
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA temp_store = MEMORY')
    cursor.execute('PRAGMA cache_size = -400000')  # ~400 MB page cache
    cursor.execute('PRAGMA page_size = 8192')      # fewer, larger writes

    # NOTE: no PRIMARY KEY here on purpose. entry_id is a random-ish TEXT key,
    # so maintaining its unique index during the load turns every insert into a
    # random read/write in a multi-GB B-tree. The index is built after the load
    # instead, where SQLite can sort once and write it out sequentially.
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            entry_id TEXT,
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


def iter_entries(xml_file):
    """Yield <entry> elements one at a time, freeing memory as we go."""
    if HAVE_LXML:
        # tag= filter means lxml only surfaces entry ends to Python.
        context = ET.iterparse(xml_file, events=('end',), tag=ENTRY_TAG,
                               huge_tree=True, recover=True)
        for _, elem in context:
            yield elem
            # Drop the element and every already-processed sibling.
            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]
    else:
        context = iter(ET.iterparse(xml_file, events=('start', 'end')))
        _, root = next(context)
        for event, elem in context:
            if event == 'end' and elem.tag == ENTRY_TAG:
                yield elem
                root.clear()


def parse_and_insert(xml_file, table_name, conn, batch_size=50000, report_every=1000000):
    """Iteratively parse the XML and batch insert records to SQLite."""
    cursor = conn.cursor()

    insert_sql = f'''
        INSERT INTO {table_name}
        (entry_id, cluster_name, member_count, common_taxon_name,
         common_taxon_id, representative_member_id, sequence_length, sequence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''

    records = []
    append = records.append
    n = 0

    for elem in iter_entries(xml_file):
        entry_id = elem.get('id')

        cluster_name = None
        member_count = None
        common_taxon_name = None
        common_taxon_id = None
        rep_member_id = None
        sequence_length = None
        sequence = None

        for child in elem:
            tag = child.tag

            if tag == PROP_TAG:
                prop_type = child.get('type')
                if prop_type == 'member count':
                    v = child.get('value')
                    member_count = int(v) if v else None
                elif prop_type == 'common taxon':
                    common_taxon_name = child.get('value')
                elif prop_type == 'common taxon ID':
                    common_taxon_id = child.get('value')

            elif tag == NAME_TAG:
                text = child.text
                if text and text.startswith('Cluster: '):
                    cluster_name = text[9:]
                else:
                    cluster_name = text

            elif tag == REP_TAG:
                for rep_child in child:
                    if rep_child.tag == DBREF_TAG:
                        rep_member_id = rep_child.get('id')
                    elif rep_child.tag == SEQ_TAG:
                        seq_len = rep_child.get('length')
                        sequence_length = int(seq_len) if seq_len else None
                        # UniRef wraps long sequences, so strip the newlines.
                        sequence = rep_child.text.replace('\n', '') if rep_child.text else None
                # The DTD puts representativeMember before every <member>, so
                # once it is handled there is nothing left to read in this entry.
                break

        append((entry_id, cluster_name, member_count,
                common_taxon_name, common_taxon_id,
                rep_member_id, sequence_length, sequence))
        n += 1

        if len(records) >= batch_size:
            cursor.executemany(insert_sql, records)
            conn.commit()
            records = []
            append = records.append

        if report_every and n % report_every == 0:
            print(f"  ...{n:,} clusters written", flush=True)

    if records:
        cursor.executemany(insert_sql, records)
        conn.commit()

    print(f"Done: {n:,} clusters.")
    return n


def finalize(conn, table_name):
    """Build the unique index after the load and restore normal durability."""
    cursor = conn.cursor()
    print("Building index on entry_id...")
    cursor.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_entry_id '
                   f'ON {table_name} (entry_id)')
    conn.commit()
    cursor.execute('PRAGMA journal_mode = DELETE')
    cursor.execute('PRAGMA synchronous = FULL')


if __name__ == '__main__':
    XML_FILE = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50.xml'
    DB_FILE = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_representatives.db'

    # XML_FILE = 'sample2000.xml'
    # DB_FILE = 'uniref50.db'

    TABLE_NAME = 'uniref50_representatives'

    print(f"Creating database {DB_FILE}...")
    print(f"XML parser: {'lxml' if HAVE_LXML else 'ElementTree (pip install lxml for ~2x)'}")
    db_conn = create_db(DB_FILE, TABLE_NAME)

    print(f"Parsing {XML_FILE} and loading to database...")
    parse_and_insert(XML_FILE, TABLE_NAME, db_conn)

    finalize(db_conn, TABLE_NAME)

    db_conn.close()
    print("Database build successfully complete!")