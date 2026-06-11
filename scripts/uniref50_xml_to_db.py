import sqlite3
import xml.etree.ElementTree as ET

def create_db(db_name="/cta/share/users/uniprot/uniref/uniref_2024_06/uniref50_representatives.db", table_name='uniref50_representatives'):
    """Initialize the SQLite database and create the table schema."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create the table with the requested columns
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

def parse_and_insert(xml_file, table_name, conn, batch_size=10000):
    """Iteratively parse the XML and batch insert records to SQLite."""
    cursor = conn.cursor()
    
    # Using iterparse to stream the XML events
    context = ET.iterparse(xml_file, events=('start', 'end'))
    context = iter(context)
    
    # Grab the root element so we can clear it dynamically to free memory
    event, root = next(context)
    
    records = []
    
    for event, elem in context:
        if event == 'end' and elem.tag.endswith('}entry'):
            # Extract basic entry attribute
            entry_id = elem.attrib.get('id')
            
            # Initialize fields
            cluster_name = None
            member_count = None
            common_taxon_name = None
            common_taxon_id = None
            rep_member_id = None
            sequence_length = None
            sequence = None
            
            # Iterate through the children of the <entry> tag
            for child in elem:
                if child.tag.endswith('}name'):
                    # Strip 'Cluster: ' prefix if it exists
                    if child.text and child.text.startswith('Cluster: '):
                        cluster_name = child.text[9:] # 9 is the length of 'Cluster: '
                    else:
                        cluster_name = child.text
                
                elif child.tag.endswith('}property'):
                    prop_type = child.attrib.get('type')
                    prop_value = child.attrib.get('value')
                    
                    if prop_type == 'member count':
                        member_count = int(prop_value) if prop_value else None
                    elif prop_type == 'common taxon':
                        common_taxon_name = prop_value
                    elif prop_type == 'common taxon ID':
                        common_taxon_id = prop_value
                        
                elif child.tag.endswith('}representativeMember'):
                    for rep_child in child:
                        if rep_child.tag.endswith('}dbReference'):
                            rep_member_id = rep_child.attrib.get('id')
                        elif rep_child.tag.endswith('}sequence'):
                            seq_len = rep_child.attrib.get('length')
                            sequence_length = int(seq_len) if seq_len else None
                            sequence = rep_child.text
                            
            # Append the extracted row to our batch
            records.append((
                entry_id, cluster_name, member_count, 
                common_taxon_name, common_taxon_id, 
                rep_member_id, sequence_length, sequence
            ))
            
            # CRITICAL: Clear the element to free up memory
            root.clear()
            
            # Once we reach the batch size, push to SQLite
            if len(records) >= batch_size:
                cursor.executemany(f'''
                    INSERT OR IGNORE INTO {table_name} 
                    (entry_id, cluster_name, member_count, common_taxon_name, 
                     common_taxon_id, representative_member_id, sequence_length, sequence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', records)
                conn.commit()
                records = []
                
    # Insert any remaining records after finishing the loop
    if records:
        cursor.executemany(f'''
            INSERT OR IGNORE INTO {table_name} 
            (entry_id, cluster_name, member_count, common_taxon_name, 
             common_taxon_id, representative_member_id, sequence_length, sequence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)
        conn.commit()

if __name__ == '__main__':
    # File paths
    XML_FILE = '/cta/share/users/uniprot/uniref/uniref_2024_06/uniref50.xml'
    DB_FILE = '/cta/share/users/uniprot/uniref/uniref_2024_06/uniref50_representatives.db'
    
    # XML_FILE = 'sample2000.xml'
    # DB_FILE = 'uniref50.db'

    TABLE_NAME = 'uniref50_representatives'
    
    print(f"Creating database {DB_FILE}...")
    db_conn = create_db(DB_FILE, TABLE_NAME)
    
    print(f"Parsing {XML_FILE} and loading to database. This will take some time...")
    parse_and_insert(XML_FILE, TABLE_NAME, db_conn)
    
    db_conn.close()
    print("Database build successfully complete!")