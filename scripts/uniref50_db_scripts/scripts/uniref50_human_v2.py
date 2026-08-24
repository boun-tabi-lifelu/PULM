import sqlite3
import pandas as pd

db_file = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_members.db'
conn = sqlite3.connect(db_file)
df_uniref50_human = pd.read_sql(f"SELECT * FROM uniref50_members WHERE taxon_id='9606'", conn)
conn.close()


db_file = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_human.db'
conn = sqlite3.connect(db_file)
df_uniref50_human.to_sql("uniref50_human_all", conn, if_exists="replace", index=False)
conn.close()

df_uniref50_human_uniprot = df_uniref50_human.dropna(subset=['uniprot_id'])

df_uniref50_human_uniprot = df_uniref50_human_uniprot[(~df_uniref50_human_uniprot['uniprot_id'].str.contains('-')) | (df_uniref50_human_uniprot['is_representative']==1)]

df_representative_members = df_uniref50_human_uniprot[df_uniref50_human_uniprot['is_representative']==1]

uniref50_human_uniprot_distilled_list = df_representative_members.to_dict('records')
representative_member_list = list(df_representative_members['representative_member'])

not_reprsented_proteins = df_uniref50_human_uniprot[df_uniref50_human_uniprot['is_representative']==0]['representative_member'].apply(lambda x: x not in representative_member_list)

distilled_members_list = df_uniref50_human_uniprot[df_uniref50_human_uniprot['is_representative']==0][not_reprsented_proteins].drop_duplicates(subset=['representative_member']).to_dict('records')

uniref50_human_uniprot_distilled_list.extend(distilled_members_list)

df_uniref50_human_uniprot_distilled = pd.DataFrame(uniref50_human_uniprot_distilled_list)

db_file = '/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_human.db'
conn = sqlite3.connect(db_file)
df_uniref50_human_uniprot_distilled.to_sql("uniref50_human_distilled_v2", conn, if_exists="replace", index=False)
conn.close()