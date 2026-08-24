import sqlite3
import xml.etree.ElementTree as ET

# Property names used inside <dbReference> elements
PROP_ACCESSION = 'UniProtKB accession'
PROP_UNIPARC = 'UniParc ID'
PROP_TAXON = 'NCBI taxonomy'


def create_db(db_name, table_name='uniref50_members'):
    """Initialize the SQLite database and create the table schema."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Bulk-load pragmas: big speedup, at the cost of crash safety during the load
    cursor.execute('PRAGMA journal_mode = OFF')
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA temp_store = MEMORY')
    cursor.execute('PRAGMA cache_size = -200000')  # ~200 MB page cache

    # No PRIMARY KEY on uniprot_id: a member may in principle appear more than
    # once, and a UNIQUE constraint would silently drop rows. Indexes are built
    # after the load instead (much faster than maintaining them during inserts).
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            uniprot_id            TEXT,
            uniprot_accession     TEXT,
            uniparc_id            TEXT,
            is_representative     INTEGER,
            representative_member TEXT,
            taxon_id              TEXT,
            common_taxon_id       TEXT
        )
    ''')
    conn.commit()
    return conn


def parse_db_reference(db_ref):
    """Pull (uniprot_id, uniprot_accession, uniparc_id, taxon_id) out of a <dbReference>."""
    ref_type = db_ref.attrib.get('type')     # 'UniProtKB ID' or 'UniParc ID'
    ref_id = db_ref.attrib.get('id')

    accession = None
    uniparc_id = None
    taxon_id = None

    for prop in db_ref:
        if not prop.tag.endswith('}property'):
            continue
        prop_type = prop.attrib.get('type')
        prop_value = prop.attrib.get('value')

        if prop_type == PROP_ACCESSION:
            accession = prop_value
        elif prop_type == PROP_UNIPARC:
            uniparc_id = prop_value
        elif prop_type == PROP_TAXON:
            taxon_id = prop_value

    # Members that only exist in UniParc have type="UniParc ID" and carry no
    # 'UniProtKB accession' property: their id is the UniParc ID itself.
    if ref_type == 'UniParc ID' and uniparc_id is None:
        uniparc_id = ref_id

    return ref_id, accession, uniparc_id, taxon_id


def parse_and_insert(xml_file, table_name, conn, batch_size=50000, report_every=100000):
    """Iteratively parse the XML and batch insert every cluster member to SQLite."""
    cursor = conn.cursor()

    insert_sql = f'''
        INSERT INTO {table_name}
        (uniprot_id, uniprot_accession, uniparc_id, is_representative,
         representative_member, taxon_id, common_taxon_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''

    # Using iterparse to stream the XML events
    context = ET.iterparse(xml_file, events=('start', 'end'))
    context = iter(context)

    # Grab the root element so we can clear it dynamically to free memory
    event, root = next(context)

    records = []
    entry_count = 0
    member_count_total = 0

    for event, elem in context:
        if event != 'end' or not elem.tag.endswith('}entry'):
            continue

        # The UniRef50 cluster id, e.g. 'UniRef50_Q8WZ42'
        cluster_id = elem.attrib.get('id')
        common_taxon_id = None

        # Entry-level properties come before representativeMember/member in the
        # DTD, so a single pass over the children is enough.
        for child in elem:
            tag = child.tag

            if tag.endswith('}property'):
                if child.attrib.get('type') == 'common taxon ID':
                    common_taxon_id = child.attrib.get('value')

            elif tag.endswith('}representativeMember') or tag.endswith('}member'):
                is_representative = 1 if tag.endswith('}representativeMember') else 0

                for sub in child:
                    if not sub.tag.endswith('}dbReference'):
                        continue  # skips <sequence> under representativeMember

                    uniprot_id, accession, uniparc_id, taxon_id = parse_db_reference(sub)

                    records.append((
                        uniprot_id, accession, uniparc_id,
                        is_representative, cluster_id,
                        taxon_id, common_taxon_id
                    ))
                    member_count_total += 1

        entry_count += 1

        # CRITICAL: Clear the element to free up memory
        root.clear()

        if len(records) >= batch_size:
            cursor.executemany(insert_sql, records)
            conn.commit()
            records = []

        if report_every and entry_count % report_every == 0:
            print(f"  ...{entry_count:,} clusters parsed, {member_count_total:,} members written",
                  flush=True)

    # Insert any remaining records after finishing the loop
    if records:
        cursor.executemany(insert_sql, records)
        conn.commit()

    print(f"Done: {entry_count:,} clusters, {member_count_total:,} members.")
    return entry_count, member_count_total


def create_indexes(conn, table_name):
    """Build lookup indexes after the bulk load (much faster than during)."""
    cursor = conn.cursor()
    print("Creating indexes (this also takes a while)...")
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_uniprot_id '
                   f'ON {table_name} (uniprot_id)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_accession '
                   f'ON {table_name} (uniprot_accession)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_uniparc '
                   f'ON {table_name} (uniparc_id)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_cluster '
                   f'ON {table_name} (representative_member)')
    conn.commit()


if __name__ == '__main__':
    # File paths
    XML_FILE = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50.xml'
    DB_FILE = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_members.db'

    # XML_FILE = 'sample2000.xml'
    # DB_FILE = 'uniref50_members.db'

    TABLE_NAME = 'uniref50_members'

    print(f"Creating database {DB_FILE}...")
    db_conn = create_db(DB_FILE, TABLE_NAME)

    print(f"Parsing {XML_FILE} and loading to database. This will take some time...")
    parse_and_insert(XML_FILE, TABLE_NAME, db_conn)

    create_indexes(db_conn, TABLE_NAME)

    db_conn.close()
    print("Database build successfully complete!")